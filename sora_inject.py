"""SoRA-style sparse gate LoRA branch."""
from __future__ import annotations
import math
from typing import List
import torch
import torch.nn as nn
from train_integration import _set_module_by_path

class SoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int, lora_alpha: float, lora_dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = float(lora_alpha) / max(r, 1)
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        d_in, d_out = base_layer.in_features, base_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        self.gate = nn.Parameter(torch.randn(1, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lora_dropout(x) @ self.lora_A.to(dtype=x.dtype).T
        h = h * self.gate.to(dtype=x.dtype)
        delta = (h @ self.lora_B.to(dtype=x.dtype).T) * self.scaling
        return self.base_layer(x) + delta

def inject_sora(model: nn.Module, target_modules: List[str], r: int, lora_alpha: float, lora_dropout: float = 0.1) -> None:
    if not target_modules:
        raise ValueError("target_modules empty")
    for p in model.parameters():
        p.requires_grad = False
    to_inject = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(sfx) for sfx in target_modules):
            to_inject.append((name, module))
    if not to_inject:
        raise ValueError("SoRA: no Linear matched")
    for name, base_linear in to_inject:
        wrapped = SoRALinear(base_linear, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout).to(base_linear.weight.device)
        _set_module_by_path(model, name, wrapped)
    for name, param in model.named_parameters():
        if "classifier" in name or "score" in name or "lm_head" in name or name == "shared" or "pooler" in name:
            param.requires_grad = True


class SparseAdamW(torch.optim.AdamW):
    """
    近端梯度 (Proximal Gradient) 的 AdamW，用于 SoRA gate 参数。

    在标准 AdamW 更新后，应用 Soft-Thresholding 对参数执行硬裁剪：
        θ ← sign(θ) · max(|θ| - λ, 0)

    这保证了门控参数可以达到精确的数学零值（稀疏性保障），相比仅在 Loss 上加 L1 惩罚的
    次梯度方法（Subgradient），近端梯度方法产生的解有更好的稀疏结构。

    参考：TsinghuaC3I/SoRA (EMNLP 2023) src/sparse_optimizer.py
    """

    def __init__(self, params, sparse_lambda: float = 1e-3, **kwargs):
        super().__init__(params, **kwargs)
        self.sparse_lambda = sparse_lambda

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        if self.sparse_lambda > 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    # Soft-thresholding: θ ← sign(θ) · max(|θ| - λ, 0)
                    p.data = torch.sign(p.data) * torch.clamp(p.data.abs() - self.sparse_lambda, min=0.0)
        return loss
