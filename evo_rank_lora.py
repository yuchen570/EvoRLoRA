import math
import torch
import torch.nn as nn
import warnings
from typing import List, Optional, Tuple

class EvoRankLoRALayer(nn.Module):
    """
    EvoRank-LoRA 的核心可演化秩层（论文 Sec 3.3 Rank Super-Space Parameterization）。

    论文公式对应：
      Eq. 100:  ΔW_ℓ = Σ_{i=1}^{R_max} m_{ℓ,i} b_{ℓ,i} a_{ℓ,i}^T
      Eq. 104:  ΔW_ℓ = B_ℓ M_ℓ A_ℓ   （M = diag(m_1, ..., m_{R_max})）
      Eq. 116:  r_ℓ = Σ m_{ℓ,i}      （有效秩 = active_mask.sum()）
      Eq. 138:  g_ℓ = ||∂L/∂ΔW_ℓ||_F  （层需求信号）
      Eq. 142:  s_{ℓ,i} = ||b_{ℓ,i}|| ||a_{ℓ,i}||  （组件重要性）
      Eq. 146:  u_ℓ = α g_ℓ + β (1/r_ℓ) Σ s_{ℓ,i}  （容量评分）
      Eq. 150:  s^red_{ℓ,i} = α₁||b||·||a|| + α₂|⟨G, b a^T⟩|  （剪枝评分）
      Prop 3.2: 最优 rank-1 扩张方向 = G 的主左右奇异向量

    实现说明：
    1. 秩超空间：lora_A (r_max × in), lora_B (out × r_max)，掩码选择活跃组件。
    2. 缩放：默认标准 LoRA (α/r)，可选 rsLoRA (α/√r, use_rslora=True)；
       expand/reduce 时做补偿 (乘 s(r_old)/s(r_new))，保持 ΔW 输出连续。
    3. 高效评价：避免实例化 G = ∂L/∂ΔW (out × in 巨矩阵)，利用 A.grad 和 B.grad
       的 trace trick 计算 g_ℓ 和 s^red_{ℓ,i}。
    4. 本层保持无状态；EMA/计数器/冷却期由 RankEvolutionController 管理。

    警告 (Zero-Gradient Trap): 由于 lora_B 零初始化，训练 Step 0 时 grad_A = B^T @ G
    必为零。外部 Controller 必须遵循论文 Sec 3.7（line 268）的 Warmup（前 10% steps
    禁止剪枝/扩张），等 B 充分更新后才能评分。
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r_max: int,
        r_init: int,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        use_rslora: bool = False,
        debug: bool = False,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r_max = r_max
        self.lora_alpha = lora_alpha
        self.use_rslora = bool(use_rslora)
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
        self._cached_prune_scores: Optional[torch.Tensor] = None
        self._cached_expand_bar_s: Optional[float] = None
        self._cached_grad_direction: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None
        
    def reset_parameters(self):
        """标准 LoRA 初始化：A Kaiming, B Zero"""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def get_active_rank(self) -> int:
        return self.active_mask.sum().item()
        
    def get_scaling_factor(self, r: int) -> float:
        """缩放因子：默认 LoRA(alpha/r)；可选 rsLoRA(alpha/sqrt(r))。"""
        rank = max(r, 1)
        if self.use_rslora:
            return self.lora_alpha / math.sqrt(rank)
        return self.lora_alpha / rank
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（论文 Eq. 100–104）：
          ΔW_ℓ = Σ_{i: m_i=1} b_i a_i^T  →  等价于 B_active @ A_active
          output = x · ΔW^T · s       （s = α/r 或 α/√r）

        实现：仅提取 active_mask=1 的行/列做矩阵乘，不实例化 R_max 维完整乘积。
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
    def activate_component(
        self,
        index: int,
        init_mode: str = "zero",
        grad_direction: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None,
    ):
        """
        激活 (Expand) 一个秩-1组件，并进行补偿归一化。

        init_mode:
            "zero"     -- A 行重置 + B 列清零（干净 cold start，默认）
            "gradient" -- 论文 Proposition 3.2：B 列和 A 行设为 ∂L/∂ΔW 的主奇异方向
        grad_direction:
            (u1, v1, sigma1) 元组，由 compute_gradient_rank1_direction() 提供。
        """
        if self.active_mask[index]:
            return

        prev_rank = self.get_active_rank()
        new_rank = prev_rank + 1

        self.active_mask[index] = True

        if init_mode == "gradient" and grad_direction is not None:
            u1, v1, _sigma1 = grad_direction
            # 用现有活跃 B 列的平均范数作为参考尺度，取 1% 作为新分量幅度
            old_indices = [i for i, m in enumerate(self.active_mask.tolist()) if m and i != index]
            if old_indices:
                idx_old = torch.as_tensor(old_indices, device=self.lora_B.weight.device, dtype=torch.long)
                avg_b_norm = torch.norm(self.lora_B.weight[:, idx_old].float(), dim=0).mean().item()
                init_scale = 0.01 * max(avg_b_norm, 1e-6)
            else:
                init_scale = 0.01
            # u1 指向 G 的主左奇异方向；取负号使 b·a^T 朝 -G 方向（降低 loss）
            self.lora_B.weight.data[:, index] = (-u1 * init_scale).to(self.lora_B.weight.dtype)
            self.lora_A.weight.data[index, :] = v1.to(self.lora_A.weight.dtype)
        else:
            nn.init.kaiming_uniform_(self.lora_A.weight[index:index + 1, :], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight[:, index])
        
        # Double Scaling 防护：应用补偿因子
        # 缩放因子从 s(prev_rank) 降为 s(new_rank)，
        # 要保持旧组件的 DeltaW = B * A 总幅值不变，需要放大旧权重。
        # 关键：只补偿 B 一侧！若同时乘到 A 和 B 上，实际效果是 c^2，会导致激活值炸飞。
        # c = s(prev) / s(new)  →  rsLoRA: sqrt(new/prev), 标准: new/prev
        if prev_rank > 0:
             compensation_factor = self.get_scaling_factor(prev_rank) / self.get_scaling_factor(new_rank)
             old_indices = [i for i, m in enumerate(self.active_mask.tolist()) if m and i != index]
             self.lora_B.weight[:, old_indices] *= compensation_factor

    @torch.no_grad()
    def deactivate_component(self, index: int):
        """
        休眠 (Prune) 一个秩-1组件（论文 Sec 3.4 Reduction）。

        Theorem 3.3（Eq. 406–411）给出移除 j-th 组件的 loss 上界：
          L(ΔW^{-j}) - L(ΔW) ≤ |⟨∇L, b_j a_j^T⟩| + (β/2)||b_j||²||a_j||²
        代码中 s^red 即为该上界的无 β 近似（Eq. 150），controller 据此判定哪些组件可裁。

        权重不清零，支持未来重激活时的权重继承（论文 Sec 3.6 "Inactive components
        keep their stored weights, which enables weight inheritance if they are reactivated"）。
        """
        if not self.active_mask[index]:
            return # 已休眠
            
        prev_rank = self.get_active_rank()
        new_rank = prev_rank - 1
        
        self.active_mask[index] = False
        
        # 反向补偿：缩放因子从 s(prev_rank) 变为 s(new_rank)（变大），
        # 留下来的组件输出会被放大。为了防止微调后期 Loss 突然毛刺，对留下来的组件乘以衰减系数对冲。
        # c = s(prev) / s(new) < 1  →  rsLoRA: sqrt(new/prev), 标准: new/prev
        if new_rank > 0:
            compensation_factor = self.get_scaling_factor(prev_rank) / self.get_scaling_factor(new_rank)
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
        self._cached_prune_scores = None
        self._cached_expand_bar_s = None
        self._cached_grad_direction = None

    @torch.no_grad()
    def cache_statistics_from_current_gradients(self, alpha1: float = 1.0, alpha2: float = 0.1):
        """
        backward 后一次性缓存 g_ℓ、s^red（剪枝）、\\bar s（扩张容量）和梯度主方向，
        避免 controller 读取时重复计算。
        """
        self._cached_demand_score = self.compute_demand_score()
        self._cached_prune_scores = self._compute_prune_scores_raw(alpha1, alpha2)
        self._cached_expand_bar_s = self._compute_expand_bar_s_raw()
        self._cached_grad_direction = self._compute_gradient_rank1_direction()

    # ------ 梯度引导的 rank-1 扩张方向 (论文 Proposition 3.2) ------

    @staticmethod
    def _power_iteration_rank1(
        N: torch.Tensor, M: torch.Tensor, num_iters: int = 5,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
        """
        论文 Proposition 3.2（Eq. 381–388）：最优 rank-1 扩张方向。

        求 G ≈ N @ M 的主奇异三元组 (u₁, v₁, σ₁)，满足
          max_{||a||=||b||=1} ⟨-G, b a^T⟩ = σ₁(G)
        其中 u₁, v₁ 分别为 G 的主左、右奇异向量。

        实现：power iteration（交替 Gv → u, G^Tu → v），每轮仅需
        O(r·(out+in)) 运算，不构造 (out, in) 的完整 G 矩阵。
        N: (out, r), M: (r, in); G = N @ M: (out, in)
        Returns (u1, v1, sigma1) or None on degenerate input.
        """
        in_dim = M.size(1)
        v = torch.randn(in_dim, device=N.device, dtype=N.dtype)
        v = v / (v.norm() + 1e-12)

        for _ in range(num_iters):
            Mv = M @ v              # (r,)
            u = N @ Mv               # (out,)
            u_norm = u.norm()
            if u_norm < 1e-12:
                return None
            u = u / u_norm

            Ntu = N.T @ u            # (r,)
            v = M.T @ Ntu            # (in,)
            v_norm = v.norm()
            if v_norm < 1e-12:
                return None
            v = v / v_norm

        Mv = M @ v
        sigma_signed = float(torch.dot(u, N @ Mv).item())
        if sigma_signed < 0:
            u = -u
        return u, v, abs(sigma_signed)

    @torch.no_grad()
    def _compute_gradient_rank1_direction(
        self,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
        """
        论文 Proposition 3.2：计算近似 G = ∂L/∂ΔW 的主 rank-1 奇异方向。
        复用 compute_demand_score 的 (A^TA+εI)^{-1}A^T 重建，然后做 power iteration。
        """
        if self.lora_A.weight.grad is None or self.lora_B.weight.grad is None:
            return None

        active_idx = self.get_active_indices()
        if len(active_idx) == 0:
            return None

        s = float(self.get_scaling_factor(len(active_idx)))
        if s == 0.0:
            return None

        idx_t = torch.as_tensor(active_idx, device=self.lora_A.weight.device, dtype=torch.long)
        gb = self.lora_B.weight.grad.index_select(1, idx_t).float()
        a = self.lora_A.weight.index_select(0, idx_t).float()

        N = gb / s                    # (out, r)
        at = a.T                      # (in, r)
        ata = at.T @ at               # (r, r)
        M = torch.linalg.solve(
            ata + 1e-6 * torch.eye(ata.size(0), device=ata.device, dtype=ata.dtype),
            at.T,
        )                             # (r, in)

        return self._power_iteration_rank1(N, M)

    @torch.no_grad()
    def compute_gradient_rank1_direction(
        self, use_cached: bool = False,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
        """获取（可缓存的）梯度主 rank-1 方向，供 ExpandMutation 使用。"""
        if use_cached and self._cached_grad_direction is not None:
            return self._cached_grad_direction
        return self._compute_gradient_rank1_direction()

    # ---- expand 路径：s_{ℓ,i} = ||b||·||a||（论文 Eq. 142） ----
    @torch.no_grad()
    def _compute_expand_bar_s_raw(self) -> float:
        """
        扩张容量项（论文 Eq. 142 + 146）：
          s_{ℓ,i} = ||b_{ℓ,i}||₂ · ||a_{ℓ,i}||₂           …Eq. 142
          s̄_ℓ = (1/max(r_ℓ,1)) Σ_{i:active} s_{ℓ,i}       …Eq. 146 中第二项

        该值进入 u_ℓ = α g̃_ℓ + β s̃̄_ℓ（Eq. 217），经 max-scaling 归一化后
        做 EMA 和分位数阈值比较。
        """
        active_idx = self.get_active_indices()
        if not active_idx:
            return 0.0
        norm_A = torch.norm(self.lora_A.weight, dim=1)
        norm_B = torch.norm(self.lora_B.weight, dim=0)
        idx_t = torch.as_tensor(active_idx, device=self.lora_A.weight.device, dtype=torch.long)
        return float((norm_A[idx_t] * norm_B[idx_t]).sum().item() / max(len(active_idx), 1))

    @torch.no_grad()
    def compute_expand_capacity_bar_s(self, use_cached: bool = False) -> float:
        """u_ℓ 中的纯范数容量均值 s̄_ℓ（论文 Eq. 146），经 controller 归一化后组合为 u_ℓ（Eq. 217）。"""
        if use_cached and self._cached_expand_bar_s is not None:
            return self._cached_expand_bar_s
        return self._compute_expand_bar_s_raw()

    # ---- prune 路径（论文 Eq. 150 + Theorem 3.3） ----
    @torch.no_grad()
    def _compute_prune_scores_raw(self, alpha1: float, alpha2: float) -> torch.Tensor:
        """
        剪枝重要性评分（论文 Eq. 150）：
          s^red_{ℓ,i} = α₁ ||b_{ℓ,i}|| · ||a_{ℓ,i}|| + α₂ |⟨∂L/∂ΔW_ℓ, b_{ℓ,i} a_{ℓ,i}^T⟩|

        与 Theorem 3.3（Eq. 406–411）的联系：
          移除组件 j 的 loss 增量上界为 |⟨∇L, b_j a_j^T⟩| + (β/2)||b_j||²||a_j||²，
          s^red 中的两项分别近似该上界的一阶项和范数项（取线性组合替代二次范数更新步幅）。
          分数小的组件对 loss 扰动小，优先候选剪枝。

        梯度交互项的计算推导（避免构造 G ∈ R^{out×in}）：
          前向：ΔW = s · B A，故 ∂L/∂A = s · B^T G（G = ∂L/∂ΔW）
          ⟨G, b_i a_i^T⟩ = Σ_j (b_i^T G)_j · a_{ij}
                          = (1/s) Σ_j grad_A[i,j] · A[i,j]
          因此 grad_interaction[i] = |Σ_j grad_A[i,j] · A[i,j]| / s

        仅活跃位有值，未激活位为 0。
        """
        if self.lora_A.weight.grad is None or self.lora_B.weight.grad is None:
            raise ValueError("需要先调用 .backward() 计算出 lora_A 和 lora_B 的梯度")
        if self.debug and not self.lora_A.weight.grad.any():
            warnings.warn("lora_A.grad 全为零，可能是 zero_grad() 之后未执行新的 backward()")

        active_idx = self.get_active_indices()
        s_layer = float(self.get_scaling_factor(max(len(active_idx), 1)))

        # |⟨G, b_i a_i⊤⟩|：rsLoRA 下 ∂L/∂A 含额外标量 s，除去后得论文量
        grad_interaction = torch.sum(self.lora_A.weight.grad * self.lora_A.weight, dim=1).abs()
        if s_layer > 0.0:
            grad_interaction = grad_interaction / s_layer

        norm_A = torch.norm(self.lora_A.weight, dim=1)
        norm_B = torch.norm(self.lora_B.weight, dim=0)
        norm_product = norm_A * norm_B

        scores = torch.zeros(self.r_max, device=self.lora_A.weight.device)
        if active_idx:
            idx_t = torch.as_tensor(active_idx, device=self.lora_A.weight.device, dtype=torch.long)
            scores[idx_t] = alpha1 * norm_product[idx_t] + alpha2 * grad_interaction[idx_t]
        return scores

    @torch.no_grad()
    def compute_prune_reduction_scores(
        self,
        alpha1: float = 1.0,
        alpha2: float = 0.1,
        use_cached: bool = False,
    ) -> torch.Tensor:
        """
        s^red_{ℓ,i}（论文 Eq. 150），供 controller 做 EMA（Eq. 234–237）与
        分位数阈值 τ_prune（Eq. 254–264）比较使用。
        """
        if use_cached and self._cached_prune_scores is not None:
            return self._cached_prune_scores
        return self._compute_prune_scores_raw(alpha1, alpha2)

    @torch.no_grad()
    def compute_component_importance(
        self,
        alpha1: float = 1.0,
        alpha2: float = 0.1,
        use_cached: bool = False,
    ) -> torch.Tensor:
        """同 compute_prune_reduction_scores（历史命名，供 controller 调用）。"""
        return self.compute_prune_reduction_scores(alpha1=alpha1, alpha2=alpha2, use_cached=use_cached)
        
    @torch.no_grad()
    def compute_demand_score(self, use_cached: bool = False) -> float:
        """
        层需求信号 g_ℓ ≈ ||∂L/∂(ΔW_ℓ)||_F（论文式 138 的投影近似）。

        注意：此实现计算的是真实 ||G||_F 的 **下界 (Lower Bound)**，而非精确值。
        原因：pinv(A^T) 给出的是最小范数解，等价于将 G 投影到 A 的行空间上；
        真实梯度矩阵 G 中正交于 A 行空间的分量被丢弃。在不使用 forward/backward
        hooks 截获激活值的前提下，这是仅凭参数梯度所能做到的最优近似。

        推导：前向为 ΔW = s · B_active A_active，故 ∂L/∂B = s G A^T、∂L/∂A = s B^T G，
        由 G A^T = (∂L/∂B)/s 得 G_proj = (∂L/∂B/s) pinv(A^T)，且
        ||G_proj||_F^2 = tr(N^T N M M^T)，其中 N = ∂L/∂B/s，M = pinv(A^T)。
        """
        if use_cached and self._cached_demand_score is not None:
            return self._cached_demand_score

        if self.lora_A.weight.grad is None or self.lora_B.weight.grad is None:
             raise ValueError("需要先调用 .backward() 计算出 lora_A 和 lora_B 的梯度")
        if self.debug and not self.lora_A.weight.grad.any():
             warnings.warn("lora_A.grad 全为零，可能是 zero_grad() 之后未执行新的 backward()")

        active_idx = self.get_active_indices()
        if len(active_idx) == 0:
            return 0.0

        s = float(self.get_scaling_factor(len(active_idx)))
        if s == 0.0:
            return 0.0

        idx_t = torch.as_tensor(active_idx, device=self.lora_A.weight.device, dtype=torch.long)
        gb = self.lora_B.weight.grad.index_select(1, idx_t).float()
        a = self.lora_A.weight.index_select(0, idx_t).float()
        n = gb / s
        at = a.T  # (in_features, r_active)
        # 正则化正规方程替代 SVD 伪逆，避免 pinv 底层 SVD 的瞬间显存峰值：
        # A^+ = (A^T A + εI)^{-1} A^T，其中 A^T A 仅为 (r, r) 极小矩阵。
        ata = at.T @ at  # (r, r)
        m = torch.linalg.solve(
            ata + 1e-6 * torch.eye(ata.size(0), device=ata.device, dtype=ata.dtype),
            at.T,
        )  # (r, in_features)
        s_mtx = n.T @ n
        t_mtx = m @ m.T
        gf_sq = torch.trace(s_mtx @ t_mtx).clamp(min=0.0)
        return float(torch.sqrt(gf_sq).item())

    def merge(self, W: nn.Parameter):
        """
        合并回基础模型权重（论文 Eq. 81: W' = W + ΔW）。
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
