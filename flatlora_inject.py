import math
import time
import torch
import torch.nn as nn

class FlatLoRAHookManager:
    """
    Flat-LoRA Hook 机制实现。

    参考论文: Flat-LoRA (ICML 2025)
    理论公式: ε ~ N(0, (γ * ρ / sqrt(n)) * ||W_i||^2)

    参数命名说明：
        flatlora_rho（本仓库参数名，对应命令行 --flatlora_rho）
            == 论文中的 σ（sigma，扰动强度系数）
        两者语义完全一致，论文推荐默认值 σ=0.05，即 --flatlora_rho 0.05。

    使用 register_forward_pre_hook 和 register_full_backward_hook
    在微批次运行的底层计算图中进行动态参数随机扰动（高斯噪声）。

    本实现的工程优势：
    1. 天然无视 Gradient Accumulation：无需通过 optimizer step 进行硬编码调度。
    2. 混合精度/混合调度原生支持：透传了基座网络 dtype/device。
    3. 极致显存控制：摒弃对全量 (m, n) 噪声 O(m*n) 保存，取而代之的是 O(m) 占用（缓存极小的 CPU 列范数）以及实时随机种子重跑还原。
    """
    def __init__(self, model: nn.Module, flatlora_rho: float, total_train_steps: int):
        self.model = model
        self.flatlora_rho = flatlora_rho
        self.total_train_steps = max(total_train_steps, 1) # 防除零
        
        self.global_step = 0
        self.seed_cache = {}
        self.device_generators = {}
        self.hooks_handles = []
        
    def update_step(self, step: int):
        self.global_step = step

    def _get_generator(self, device):
        if device not in self.device_generators:
            self.device_generators[device] = torch.Generator(device=device)
        return self.device_generators[device]

    def _flatlora_pre_forward_hook(self, module, args):
        if not module.training:
            return args

        step = self.global_step
        # 严格执行: cosine-increasing 调度策略（平缓增大随机性）
        factor = 0.5 * (1 - math.cos(step / self.total_train_steps * math.pi))
        
        module_idx = getattr(module, "_flatlora_module_idx", 0)
        cur_seed = (step * 9973 + module_idx) % (2**32 - 1)

        # 获取底层基线矩阵 W
        if hasattr(module, "base_layer"):
            md = module.base_layer.weight
        else:
            md = module.weight

        with torch.no_grad():
            if isinstance(module.lora_A, (dict, nn.ModuleDict)):
                adapter_name = "default" if "default" in module.lora_A else next(iter(module.lora_A.keys()))
                weight_A = module.lora_A[adapter_name].weight
                weight_B = module.lora_B[adapter_name].weight
                scaling = module.scaling[adapter_name]
            else:
                weight_A = module.lora_A
                weight_B = module.lora_B
                scaling = module.scaling

            # W' = W + s * BA
            data_equiv = md.data + scaling * (weight_B @ weight_A)
            
            # W'_{i,:} 的二范数计算
            norm_sq = torch.norm(data_equiv.float(), dim=1, keepdim=True).to(data_equiv.dtype)
            filter_norm = (factor * (self.flatlora_rho + 1e-16) / math.sqrt(data_equiv.shape[1]) * norm_sq).nan_to_num(0.0).clamp(min=1e-8)

            # 使用独立 Generator 生成噪声，避免干扰模型内的 Dropout
            gen = self._get_generator(md.device)
            gen.manual_seed(cur_seed)
            tmp = torch.normal(mean=0.0, std=filter_norm.repeat(1, md.shape[1]), generator=gen).to(dtype=md.dtype)
            
            # 记录原始的高精度权重
            self.seed_cache[id(module)] = md.data
            md.data = md.data + tmp
            
        return args

    def _flatlora_post_forward_hook(self, module, inputs, output):
        # Forward执行完后立刻恢复基网络权重，避免污染且无需依赖不可靠的 Backward Hook
        if not module.training:
            return output
        if id(module) in self.seed_cache:
            original_data = self.seed_cache.pop(id(module))
            if hasattr(module, "base_layer"):
                module.base_layer.weight.data = original_data
            else:
                module.weight.data = original_data
        return output

    def attach_hooks(self):
        """遍历模型寻找所有的 LoraLinear，并植入前向与反向恢复钩子。"""
        module_idx = 0
        for n, module in self.model.named_modules():
            if type(module).__name__ == "Linear" and hasattr(module, "lora_A"):
                module._flatlora_module_idx = module_idx
                h1 = module.register_forward_pre_hook(self._flatlora_pre_forward_hook)
                # 使用 forward_hook 恢复，确保在 loss 产生前完美恢复干净状态
                h2 = module.register_forward_hook(self._flatlora_post_forward_hook)
                self.hooks_handles.extend([h1, h2])
                module_idx += 1

    def remove_hooks(self):
        """安全撤销注入，防止脏上下文。"""
        for h in self.hooks_handles:
            h.remove()
        self.hooks_handles.clear()
        self.seed_cache.clear()
