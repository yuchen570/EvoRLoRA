import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class FlatLoRAHookManager:
    """
    Flat-LoRA 扰动管理器（显式调用版）。

    生命周期：
    1) prepare_step(global_step)
    2) perturb_before_forward()
    3) restore_after_backward()

    说明：
    - 同一个 global_step 内，多次 perturb 会使用同一 seed 规则，保证噪声一致。
    - 兼容梯度累积：只要 global_step 不变，micro-batch 将得到同分布且可复现的噪声。
    """

    def __init__(self, model: nn.Module, flatlora_rho: float, total_train_steps: int):
        self.model = model
        self.flatlora_rho = float(flatlora_rho)
        self.total_train_steps = max(int(total_train_steps), 1)
        self.global_step = 0

        self.device_generators: Dict[torch.device, torch.Generator] = {}
        self._target_modules: List[nn.Module] = []
        self._perturb_cache: Dict[int, Dict[str, object]] = {}
        self._is_perturbed = False

        self._collect_target_modules()

    def _collect_target_modules(self) -> None:
        self._target_modules.clear()
        module_idx = 0
        for _, module in self.model.named_modules():
            if type(module).__name__ == "Linear" and hasattr(module, "lora_A"):
                module._flatlora_module_idx = module_idx
                self._target_modules.append(module)
                module_idx += 1

    def _get_generator(self, device: torch.device) -> torch.Generator:
        if device not in self.device_generators:
            self.device_generators[device] = torch.Generator(device=device)
        return self.device_generators[device]

    @staticmethod
    def _get_base_weight(module: nn.Module) -> torch.Tensor:
        if hasattr(module, "base_layer"):
            return module.base_layer.weight
        return module.weight

    @staticmethod
    def _get_lora_factors(module: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, float]:
        if isinstance(module.lora_A, (dict, nn.ModuleDict)):
            adapter_name = "default" if "default" in module.lora_A else next(iter(module.lora_A.keys()))
            weight_A = module.lora_A[adapter_name].weight
            weight_B = module.lora_B[adapter_name].weight
            scaling = float(module.scaling[adapter_name])
        else:
            weight_A = module.lora_A
            weight_B = module.lora_B
            scaling = float(module.scaling)
        return weight_A, weight_B, scaling

    def prepare_step(self, step: int) -> None:
        self.global_step = int(step)

    # backward-compatible alias
    def update_step(self, step: int) -> None:
        self.prepare_step(step)

    def perturb_before_forward(self) -> None:
        if self._is_perturbed or self.flatlora_rho <= 0:
            return

        step = self.global_step
        factor = 0.5 * (1 - math.cos(step / self.total_train_steps * math.pi))
        self._perturb_cache.clear()

        with torch.no_grad():
            for module in self._target_modules:
                if not module.training:
                    continue

                module_idx = getattr(module, "_flatlora_module_idx", 0)
                cur_seed = (step * 9973 + module_idx) % (2**32 - 1)

                md = self._get_base_weight(module)
                weight_A, weight_B, scaling = self._get_lora_factors(module)
                data_equiv = md.data + scaling * (weight_B @ weight_A)

                row_norm = torch.norm(data_equiv.float(), dim=1, keepdim=True).to(data_equiv.dtype)
                filter_norm = (
                    factor
                    * (self.flatlora_rho + 1e-16)
                    / math.sqrt(data_equiv.shape[1])
                    * row_norm
                ).nan_to_num(0.0).clamp(min=1e-8)

                gen = self._get_generator(md.device)
                gen.manual_seed(cur_seed)
                noise = torch.normal(
                    mean=0.0,
                    std=filter_norm.repeat(1, md.shape[1]),
                    generator=gen,
                ).to(dtype=md.dtype)
                md.data.add_(noise)

                self._perturb_cache[id(module)] = {
                    "seed": int(cur_seed),
                    "filter_norm_cpu": filter_norm.detach().cpu(),
                }

        self._is_perturbed = True

    def restore_after_backward(self) -> None:
        if not self._is_perturbed:
            return

        with torch.no_grad():
            for module in self._target_modules:
                if id(module) not in self._perturb_cache:
                    continue
                if not module.training:
                    continue

                info = self._perturb_cache[id(module)]
                md = self._get_base_weight(module)
                filter_norm = info["filter_norm_cpu"].to(device=md.device, dtype=md.dtype)
                seed = int(info["seed"])

                gen = self._get_generator(md.device)
                gen.manual_seed(seed)
                noise = torch.normal(
                    mean=0.0,
                    std=filter_norm.repeat(1, md.shape[1]),
                    generator=gen,
                ).to(dtype=md.dtype)
                md.data.sub_(noise)

        self._perturb_cache.clear()
        self._is_perturbed = False

    # backward-compatible no-op API
    def attach_hooks(self) -> None:
        return

    def remove_hooks(self) -> None:
        self.restore_after_backward()
        self._perturb_cache.clear()
