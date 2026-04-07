import math
import time
import torch
import torch.nn as nn

class FlatLoRAHookManager:
    """
    Flat-LoRA Hook 机制实现。
    
    参考论文: Flat-LoRA (ICML 2025)
    理论公式: ε ~ N(0, (γ * ρ / sqrt(n)) * ||W_i||^2)
    
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
        self.hooks_handles = []

    def update_step(self, global_step: int):
        """主训练循环同步调用该方法来广播当前的训练步（挂钩调度器）。"""
        self.global_step = global_step

    def _flatlora_pre_forward_hook(self, module, args):
        if not module.training:
            return args

        step = self.global_step
        # 严格执行: cosine-increasing 调度策略（平缓增大随机性）
        factor = 0.5 * (1 - math.cos(step / self.total_train_steps * math.pi))
        
        # 使用 global_step 和固定的 module_idx 共同构造确定性的随机种子，
        # 既能保证不同层之间的噪声相互独立，又能确保在多卡 DDP 训练时各卡的扰动严格一致，避免梯度发散。
        module_idx = getattr(module, "_flatlora_module_idx", 0)
        cur_seed = (step * 9973 + module_idx) % (2**32 - 1)

        # 获取底层基线矩阵 W
        if hasattr(module, "base_layer"):
            md = module.base_layer.weight
        else:
            md = module.weight

        with torch.no_grad():
            # 兼容 Huggingface PEFT 不同的版本抽象 (如 dict 与 ModuleDict)
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
            
            # W'_{i,:} 的二范数计算（针对 out_features 进行 Row-wise/Filter-wise 计算）
            norm_sq = torch.norm(data_equiv, dim=1, keepdim=True)
            filter_norm = factor * (self.flatlora_rho + 1e-16) / math.sqrt(data_equiv.shape[1]) * norm_sq

            # 重置随机状态，使用统一 dtype/device
            torch.manual_seed(cur_seed)
            tmp = torch.normal(0, filter_norm.repeat(1, md.shape[1])).to(dtype=md.dtype, device=md.device)
            md.data += tmp

            # 把极低的负担移交给 CPU 侧的字典中
            self.seed_cache[id(module)] = (cur_seed, filter_norm.cpu())
            
        return args

    def _flatlora_post_backward_hook(self, module, grad_input, grad_output):
        if not module.training:
            return

        if id(module) in self.seed_cache:
            cur_seed, filter_norm_cpu = self.seed_cache.pop(id(module))

            if hasattr(module, "base_layer"):
                md = module.base_layer.weight
            else:
                md = module.weight

            with torch.no_grad():
                # 重新应用相同的种子重建随机矩阵进行还原
                torch.manual_seed(cur_seed)
                filter_norm = filter_norm_cpu.to(device=md.device, dtype=md.dtype)
                tmp = torch.normal(0, filter_norm.repeat(1, md.shape[1])).to(dtype=md.dtype, device=md.device)
                
                md.data -= tmp

    def attach_hooks(self):
        """遍历模型寻找所有的 LoraLinear，并植入前向与反向钩子。"""
        module_idx = 0
        for n, module in self.model.named_modules():
            if type(module).__name__ == "Linear" and hasattr(module, "lora_A"):
                module._flatlora_module_idx = module_idx
                h1 = module.register_forward_pre_hook(self._flatlora_pre_forward_hook)
                h2 = module.register_full_backward_hook(self._flatlora_post_backward_hook)
                self.hooks_handles.extend([h1, h2])
                module_idx += 1

    def remove_hooks(self):
        """安全撤销注入，防止脏上下文。"""
        for h in self.hooks_handles:
            h.remove()
        self.hooks_handles.clear()
        self.seed_cache.clear()
