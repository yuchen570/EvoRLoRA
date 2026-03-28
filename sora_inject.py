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
        h = self.lora_dropout(x) @ self.lora_A.T
        h = h * self.gate
        delta = (h @ self.lora_B.T) * self.scaling
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
        if "classifier" in name or "score" in name or "lm_head" in name or name == "shared":
            param.requires_grad = True
