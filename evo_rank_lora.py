import math
import torch
import torch.nn as nn
import warnings
from typing import List, Optional

class EvoRankLoRALayer(nn.Module):
    """
    EvoRank-LoRA 的核心可演化秩层。
    
    论文机制实现:
    1. 秩超空间 (Rank Super-Space): 维护最大秩为 r_max 的 A 和 B 矩阵。
    2. 掩码控制 (Mask Control): 使用二进制掩码激活部分秩-1组件, 未激活部分不参与前向计算。
    3. 缩放平滑 (Scaling & Smoothing): 遵循 rsLoRA (lora_alpha / sqrt(r)), 并在扩张/缩减时进行权重补偿。 
    4. 高效评价 (Trace Trick): 避免实例化高显存占用的梯度矩阵 G = dL/dW_l, 利用 A.grad 和 B.grad 进行评分。
    
    注意: 本层保持"无状态"(Stateless)，仅负责响应外部控制。
    EMA历史、计数器 H_g/H_p、冷却期控制应由外部 Controller 负责。
    
    警告 (Zero-Gradient Trap): 由于 lora_B 零初始化，训练 Step 0 时 grad_A = B^T @ G 必为零，
    此时调用 compute_component_importance 会返回全零评分。外部 Controller 必须严格执行
    论文规定的 Warmup 机制（前 10% steps 禁止剪枝/扩张），等 B 充分更新后才能评分。
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r_max: int,
        r_init: int,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        debug: bool = False,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r_max = r_max
        self.lora_alpha = lora_alpha
        self.debug = debug  # 调试模式：开启后会检查全零梯度（触发 GPU->CPU 同步，DDP 下慎用）
        
        if r_init > r_max:
            raise ValueError("r_init 必须小于或等于 r_max")
            
        # PEFT 兼容的命名方式: lora_A, lora_B
        self.lora_A = nn.Linear(in_features, r_max, bias=False)
        self.lora_B = nn.Linear(r_max, out_features, bias=False)
        
        # Dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
            
        # 掩码状态字典 (0/1), 不参与梯度计算
        mask_init = torch.zeros(r_max, dtype=torch.bool)
        mask_init[:r_init] = True
        self.register_buffer("active_mask", mask_init)

        # 参数初始化
        self.reset_parameters()
        # 反向传播后可缓存的统计量（避免同一步重复计算）
        self._cached_demand_score: Optional[float] = None
        self._cached_component_importance: Optional[torch.Tensor] = None
        
    def reset_parameters(self):
        """标准 LoRA 初始化：A Kaiming, B Zero"""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def get_active_rank(self) -> int:
        return self.active_mask.sum().item()
        
    def get_scaling_factor(self, r: int) -> float:
        """rsLoRA 风格的缩放因子: alpha / sqrt(r)"""
        return self.lora_alpha / math.sqrt(max(r, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算:
        仅提取 current active components，避免在零组件上进行冗余矩阵乘法。
        """
        active_indices = self.get_active_indices()
        
        # 若所有组件都被 de-activate（通常由 r_min 控制避免发生），直接返回 0
        if len(active_indices) == 0:
            return torch.zeros((x.size(0), self.out_features), device=x.device, dtype=x.dtype)
            
        x = self.lora_dropout(x)
        
        # 仅截取激活维度的向量:
        # lora_A.weight 形状: (r_max, in_features)
        A_active = self.lora_A.weight[active_indices, :].to(x.dtype)
        # lora_B.weight 形状: (out_features, r_max)
        B_active = self.lora_B.weight[:, active_indices].to(x.dtype)
        
        # 计算低秩投影
        out = x @ A_active.T @ B_active.T
        
        # rsLoRA 缩放
        scaling = self.get_scaling_factor(len(active_indices))
        return out * scaling

    @torch.no_grad()
    def activate_component(self, index: int):
        """
        激活 (Expand) 一个秩-1组件，并进行补偿归一化。
        """
        if self.active_mask[index]:
            return # 已激活

        prev_rank = self.get_active_rank()
        new_rank = prev_rank + 1
        
        self.active_mask[index] = True
        
        # 新激活的 B 列不应绝对为 0，这可能阻碍梯度流动
        # 使用小的高斯噪声进行初始化
        nn.init.normal_(self.lora_B.weight[:, index], mean=0.0, std=1e-5)
        
        # Double Scaling 防护：应用补偿因子
        # 缩放因子从 alpha/sqrt(prev_rank) 降为 alpha/sqrt(new_rank)，
        # 要保持旧组件的 DeltaW = B * A 总幅值不变，需要放大旧权重。
        # 关键：只补偿 B 一侧！若同时乘到 A 和 B 上，实际效果是 c^2，会导致激活值炸飞。
        # c = sqrt(new_rank / prev_rank)
        if prev_rank > 0:
             compensation_factor = math.sqrt(new_rank / prev_rank)
             old_indices = [i for i, m in enumerate(self.active_mask.tolist()) if m and i != index]
             # 仅补偿 B，绝对不能同时补偿 A 和 B！
             self.lora_B.weight[:, old_indices] *= compensation_factor

    @torch.no_grad()
    def deactivate_component(self, index: int):
        """
        休眠 (Prune) 一个秩-1组件。
        权重不清零，支持未来重激活时的权重继承 (Weight Inheritance)。
        """
        if not self.active_mask[index]:
            return # 已休眠
            
        prev_rank = self.get_active_rank()
        new_rank = prev_rank - 1
        
        self.active_mask[index] = False
        
        # 反向补偿：rsLoRA 缩放因子从 alpha/sqrt(prev_rank) 变为 alpha/sqrt(new_rank)（变大），
        # 留下来的组件输出会被放大约 sqrt(prev/new) 倍（例如 5->4 时放大 ~11.8%）。
        # 为了防止微调后期 Loss 突然毛刺，对留下来的组件乘以衰减系数进行对冲。
        # c = sqrt(new_rank / prev_rank) < 1
        if new_rank > 0:
            compensation_factor = math.sqrt(new_rank / prev_rank)
            remaining_indices = self.get_active_indices()  # index 已被去激活
            self.lora_B.weight[:, remaining_indices] *= compensation_factor
        
    def get_active_indices(self) -> List[int]:
        """使用原生 nonzero() 避免 CPU-GPU 同步和列表转换开销"""
        return self.active_mask.nonzero(as_tuple=True)[0].tolist()
        
    def get_inactive_indices(self) -> List[int]:
        """返回所有未激活组件的索引"""
        return (~self.active_mask).nonzero(as_tuple=True)[0].tolist()

    # ------ 高效评价机制 (Trace Trick) ------
    @torch.no_grad()
    def clear_statistics_cache(self):
        """清空当前步的缓存统计量。"""
        self._cached_demand_score = None
        self._cached_component_importance = None

    @torch.no_grad()
    def cache_statistics_from_current_gradients(self, alpha1: float = 1.0, alpha2: float = 0.1):
        """
        在 backward 之后立即缓存本步统计量。
        这样 controller 在结构步骤读取分数时无需重复计算。
        """
        self._cached_demand_score = self.compute_demand_score()
        self._cached_component_importance = self.compute_component_importance(alpha1=alpha1, alpha2=alpha2)

    @torch.no_grad()
    def compute_component_importance(
        self,
        alpha1: float = 1.0,
        alpha2: float = 0.1,
        use_cached: bool = False,
    ) -> torch.Tensor:
        """
        计算活跃组件的重要性评分 (Importance Score s_{l,i})
        完全避免实例化 G (d x k 大小的梯度矩阵)。
        """
        if use_cached and self._cached_component_importance is not None:
            return self._cached_component_importance

        # 鲁棒性检查
        if self.lora_A.weight.grad is None or self.lora_B.weight.grad is None:
             raise ValueError("需要先调用 .backward() 计算出 lora_A 和 lora_B 的梯度")
        # .any() 会触发 Device-to-Host 隐式同步，DDP 下频繁调用会拖慢性能，仅 debug 模式开启
        if self.debug and not self.lora_A.weight.grad.any():
             warnings.warn("lora_A.grad 全为零，可能是 zero_grad() 之后未执行新的 backward()")
             
        # s_1: 梯度交互项 | <G, b_i a_i^T> |
        # lora_A.grad 形状: (r_max, in_features), lora_A.weight 形状: (r_max, in_features)
        # 通过在维度 1 (in_features) 上逐元素相乘后求和，等效提取出迹:
        grad_interaction = torch.sum(self.lora_A.weight.grad * self.lora_A.weight, dim=1).abs()
        # [可选等价实现]
        # grad_interaction = torch.sum(self.lora_B.weight.grad * self.lora_B.weight, dim=0).abs()
        
        # s_2: 范数项 ||b_i|| * ||a_i||
        norm_A = torch.norm(self.lora_A.weight, dim=1) # (r_max,)
        norm_B = torch.norm(self.lora_B.weight, dim=0) # (r_max,)
        norm_product = norm_A * norm_B
        
        # 总分，仅计算 active indices，未激活的返回 0.0
        scores = torch.zeros(self.r_max, device=self.lora_A.weight.device)
        active_idx = self.get_active_indices()
        
        scores[active_idx] = alpha1 * grad_interaction[active_idx] + alpha2 * norm_product[active_idx]
        return scores
        
    @torch.no_grad()
    def compute_demand_score(self, use_cached: bool = False) -> float:
        """
        计算当前层的容量需求分数代理 (Demand Score g_l)
        用 ||grad_A||_F + ||grad_B||_F 这个代理指标完美承担相对大小的比较任务。
        """
        if use_cached and self._cached_demand_score is not None:
            return self._cached_demand_score

        if self.lora_A.weight.grad is None or self.lora_B.weight.grad is None:
             raise ValueError("需要先调用 .backward() 计算出 lora_A 和 lora_B 的梯度")
        if self.debug and not self.lora_A.weight.grad.any():
             warnings.warn("lora_A.grad 全为零，可能是 zero_grad() 之后未执行新的 backward()")
             
        grad_norm_A = torch.norm(self.lora_A.weight.grad, p='fro')
        grad_norm_B = torch.norm(self.lora_B.weight.grad, p='fro')
        return (grad_norm_A + grad_norm_B).item()

    def merge(self, W: nn.Parameter):
        """
        合并回基础模型权重。
        W: 形状为 (out_features, in_features)
        """
        if self.get_active_rank() == 0:
            return
            
        active_indices = self.get_active_indices()
        A_active = self.lora_A.weight[active_indices, :].to(W.dtype)
        B_active = self.lora_B.weight[:, active_indices].to(W.dtype)
        scaling = self.get_scaling_factor(len(active_indices))
        
        delta_W = (B_active @ A_active) * scaling
        W.data += delta_W
        
    def extra_repr(self) -> str:
        r_active = self.get_active_rank()
        return f"in_features={self.in_features}, out_features={self.out_features}, r_max={self.r_max}, active={r_active}, alpha={self.lora_alpha}"
