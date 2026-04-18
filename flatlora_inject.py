"""
Flat-LoRA：在全参数视角下对「合并权重」施加 filter-wise 随机高斯扰动（贝叶斯期望目标）。

论文：Li et al., ICML 2025, Eq. (7) —— 对合并矩阵 W' = W + BA（文中省略 LoRA 缩放 s=α/r，可并入 A、B）
    (ε_W)_{i,j} ~ N(0, (σ²/n) · ||W'_{i,:}||²_2)
其中 n 为输入维，σ 为扰动强度（脚本参数 --flatlora_rho），并配合从 0→1 的余弦调度因子
    factor(t) = 0.5 · (1 - cos(π t / T))
与论文 4.1 节「cosine-increasing strategy」一致。

参考实现： https://github.com/nblt/Flat-LoRA/blob/main/logTrainer.py 中 FlatLoraTrainer
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from peft.tuners.lora.layer import Linear as PeftLoraLinear
except ImportError:
    PeftLoraLinear = None  # type: ignore[misc, assignment]


def _is_injected_lora_linear(module: nn.Module) -> bool:
    if PeftLoraLinear is not None and isinstance(module, PeftLoraLinear):
        return True
    return type(module).__name__ == "Linear" and hasattr(module, "lora_A") and hasattr(module, "lora_B")


class FlatLoRAHookManager:
    """
    Flat-LoRA 扰动管理器（显式调用版，对齐论文 Eq.(7) 与 nblt/Flat-LoRA FlatLoraTrainer）。

    生命周期：
    1) prepare_step(global_step)
    2) perturb_before_forward()
    3) restore_after_backward()
    """

    def __init__(
        self,
        model: nn.Module,
        flatlora_rho: float,
        total_train_steps: int,
        is_main_process: bool = True,
        local_rank: int = 0,
    ):
        self.model = model
        self.flatlora_rho = float(flatlora_rho)
        self.total_train_steps = max(int(total_train_steps), 1)
        self.global_step = 0
        self._is_main = is_main_process
        self._local_rank = local_rank

        self.device_generators: Dict[torch.device, torch.Generator] = {}
        self._target_modules: List[nn.Module] = []
        self._perturb_cache: Dict[int, Dict[str, object]] = {}
        self._is_perturbed = False
        self._weight_norms_before: Dict[int, float] = {}

        self._collect_target_modules()

        if self._is_main:
            print(f"[Flat-LoRA][init] rho={self.flatlora_rho}, T={self.total_train_steps}, "
                  f"target_modules={len(self._target_modules)}")
            if self._target_modules:
                m0 = self._target_modules[0]
                w0 = self._get_base_weight(m0)
                print(f"[Flat-LoRA][init] 首层 weight: shape={tuple(w0.shape)}, dtype={w0.dtype}, "
                      f"device={w0.device}, requires_grad={w0.requires_grad}, "
                      f"norm={w0.data.float().norm().item():.4f}")
            else:
                print("[Flat-LoRA][WARNING] 未收集到任何 LoRA 注入层！噪声不会生效！")

    def _collect_target_modules(self) -> None:
        self._target_modules.clear()
        module_idx = 0
        for _, module in self.model.named_modules():
            if _is_injected_lora_linear(module):
                module._flatlora_module_idx = module_idx  # noqa: SLF001
                self._target_modules.append(module)
                module_idx += 1

    def _get_generator(self, device: torch.device) -> torch.Generator:
        if device not in self.device_generators:
            self.device_generators[device] = torch.Generator(device=device)
        return self.device_generators[device]

    @staticmethod
    def _get_base_weight(module: nn.Module) -> torch.Tensor:
        """与官方 FlatLoraTrainer 一致使用 module.weight（冻结的预训练权重）。"""
        w = getattr(module, "weight", None)
        if w is not None:
            return w
        if hasattr(module, "base_layer"):
            return module.base_layer.weight
        raise AttributeError("Flat-LoRA: 无法解析 Linear 权重")

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

    @classmethod
    def _merged_weight(cls, module: nn.Module) -> torch.Tensor:
        """W' = W + s · B A，与论文 §3.3 及官方 logTrainer 中 data 一致。"""
        md = cls._get_base_weight(module)
        weight_A, weight_B, scaling = cls._get_lora_factors(module)
        return md.data + scaling * (weight_B @ weight_A)

    def _should_log(self, step: Optional[int] = None) -> bool:
        s = step if step is not None else self.global_step
        if s <= 2:
            return True  # DDP 调试：前 3 步所有 rank 均输出，便于对比跨卡一致性
        return self._is_main and (s % 100 == 0)

    def prepare_step(self, step: int) -> None:
        self.global_step = int(step)

    def update_step(self, step: int) -> None:
        self.prepare_step(step)

    def perturb_before_forward(self) -> None:
        if self._is_perturbed or self.flatlora_rho <= 0:
            return

        step = self.global_step
        t_ratio = float(step) / float(self.total_train_steps)
        factor = 0.5 * (1.0 - math.cos(t_ratio * math.pi))
        self._perturb_cache.clear()
        self._weight_norms_before.clear()
        do_log = self._should_log(step)

        noise_abs_sum = 0.0
        noise_count = 0
        modules_perturbed = 0

        with torch.no_grad():
            for module in self._target_modules:
                if not module.training:
                    continue

                module_idx = getattr(module, "_flatlora_module_idx", 0)
                
                # 官方 FlatLoraTrainer 使用 time.time() 作 seed，每 rank 独立采样（DDP 下不同 rank 扰动不同），
                # all-reduce 后的梯度近似 SAM 的 per-worker 平均。此处按 (step, module_idx, local_rank)
                # 确定性生成 seed：既保证可复现，又保留 per-rank 扰动多样性。
                cur_seed = (step * 9973 + module_idx * 131 + self._local_rank * 7919 + 1) % (2**32 - 1)

                md = self._get_base_weight(module)
                if do_log:
                    self._weight_norms_before[id(module)] = float(md.data.float().norm().item())

                data_equiv = self._merged_weight(module)
                n_in = int(data_equiv.shape[1])
                row_norm = torch.norm(data_equiv.float(), dim=1, keepdim=True).to(dtype=data_equiv.dtype)
                sigma = self.flatlora_rho + 1e-16
                filter_norm = (factor * sigma / math.sqrt(float(n_in)) * row_norm).nan_to_num(0.0).clamp(min=1e-8)

                gen = self._get_generator(md.device)
                gen.manual_seed(int(cur_seed))
                noise = torch.normal(
                    mean=0.0,
                    std=filter_norm.repeat(1, md.shape[1]),
                    generator=gen,
                ).to(dtype=md.dtype)
                md.data.add_(noise)

                if do_log:
                    noise_abs_sum += float(noise.abs().sum().item())
                    noise_count += noise.numel()

                self._perturb_cache[id(module)] = {
                    "seed": int(cur_seed),
                    "filter_norm_cpu": filter_norm.detach().cpu(),
                }
                modules_perturbed += 1

        self._is_perturbed = True

        if do_log:
            noise_abs_mean = noise_abs_sum / max(noise_count, 1)
            w0_after = float(self._get_base_weight(self._target_modules[0]).data.float().norm().item()) if self._target_modules else 0
            w0_before = self._weight_norms_before.get(id(self._target_modules[0]), 0) if self._target_modules else 0
            print(
                f"[Flat-LoRA][perturb] rank={self._local_rank} step={step} factor={factor:.6f} "
                f"modules={modules_perturbed} noise_abs_mean={noise_abs_mean:.6e} "
                f"w0_norm: {w0_before:.4f}->{w0_after:.4f} (delta={w0_after - w0_before:+.6f})"
            )

    def restore_after_backward(self) -> None:
        if not self._is_perturbed:
            return

        do_log = self._should_log()
        restore_count = 0

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
                restore_count += 1

        if do_log and self._target_modules:
            w0_restored = float(self._get_base_weight(self._target_modules[0]).data.float().norm().item())
            w0_before = self._weight_norms_before.get(id(self._target_modules[0]), 0)
            residual = abs(w0_restored - w0_before)
            status = "OK" if residual < 1e-4 else f"MISMATCH(residual={residual:.6e})"
            print(
                f"[Flat-LoRA][restore] rank={self._local_rank} step={self.global_step} "
                f"restored={restore_count} w0_norm={w0_restored:.4f} "
                f"expected={w0_before:.4f} [{status}]"
            )

        self._perturb_cache.clear()
        self._weight_norms_before.clear()
        self._is_perturbed = False

    def attach_hooks(self) -> None:
        return

    def remove_hooks(self) -> None:
        self.restore_after_backward()
        self._perturb_cache.clear()
