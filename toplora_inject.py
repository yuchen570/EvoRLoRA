"""TopLoRA: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation (NeurIPS 2025).

参考实现: toplora-neurips25/mypeft/toplora.py
论文: Li et al., "Beyond Higher Rank: Token-wise Input-Output Projections
      for Efficient Low-Rank Adaptation", NeurIPS 2025.

核心思想: 在标准 LoRA 的低秩空间中引入 token-dependent 的奇异值缩放，
         使每个 token 拥有独立的 rank 激活强度，等效于在固定秩预算下
         实现更灵活的逐 token 低秩近似。

标准 LoRA:   result = W(x) + B(A(dropout(x))) * scaling
TopLoRA:     result = W(x) + B(A(dropout(x)) * λ(x)) * scaling

其中 λ(x) = exp(RMSNorm(x @ W_λ))，W_λ ∈ R^{d_in × r}
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn

from train_integration import _set_module_by_path


class TopSingularValue(nn.Module):
    """Per-token singular value gating.

    对输入 token 映射到 r 维正缩放向量：
        λ(x) = exp(RMSNorm(x @ W_λ))

    - W_λ: (d_in, r)，Kaiming fan_out 初始化
    - RMSNorm: 对 r 维做归一化，稳定训练
    - exp: 保证输出严格正值
    """

    def __init__(self, token_dim: int, r: int, dtype=None):
        super().__init__()
        self.r = r
        self.token_dim = token_dim
        self.weight = nn.Parameter(torch.empty((token_dim, r), dtype=dtype))
        self.rms_norm = nn.RMSNorm([r])
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), mode="fan_out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = x @ self.weight
        weight = self.rms_norm(weight)
        weight = torch.exp(weight)
        return weight


class TopLoRALinear(nn.Module):
    """LoRA with token-wise singular value scaling (TopLoRA).

    前向:
        h = dropout(x) @ A^T          # (*, r)
        λ = TopSingularValue(x)        # (*, r)
        delta = (h * λ) @ B^T * scaling
        output = base_layer(x) + delta
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int,
        lora_alpha: float,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.r = r
        self.scaling = float(lora_alpha) / max(r, 1)

        self.lora_dropout = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        )

        d_in = base_layer.in_features
        d_out = base_layer.out_features

        # LoRA 低秩矩阵
        self.lora_A = nn.Parameter(torch.zeros(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))

        # TopLoRA 核心: token-wise singular value gating
        self.lora_lambda = TopSingularValue(token_dim=d_in, r=r)

        # 初始化: A 使用 Kaiming，B 清零（保证初始 delta=0）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)

        # TopLoRA forward
        x_lora = x.to(self.lora_A.dtype)
        singular_values = self.lora_lambda(x_lora)          # (*, r)
        h = self.lora_dropout(x_lora) @ self.lora_A.T       # (*, r)
        h = h * singular_values                              # token-wise scaling
        delta = (h @ self.lora_B.T) * self.scaling           # (*, d_out)

        return result + delta.to(result.dtype)


def inject_toplora(
    model: nn.Module,
    target_modules: List[str],
    r: int,
    lora_alpha: float,
    lora_dropout: float = 0.05,
) -> None:
    """将命中后缀的 nn.Linear 替换为 TopLoRALinear。

    与 sora_inject.inject_sora 同构模式：
    1. 冻结全部基座参数
    2. 按 target_modules 后缀匹配 nn.Linear
    3. 替换为 TopLoRALinear
    4. 解冻分类头等可训练参数

    Args:
        model: 待注入模型
        target_modules: 目标后缀列表，如 ["query_proj", "key_proj", "value_proj"]
        r: LoRA 秩
        lora_alpha: LoRA alpha 缩放系数
        lora_dropout: TopLoRA dropout（论文默认 0.05）
    """
    if not target_modules:
        raise ValueError("target_modules empty")

    # 冻结全部基座参数
    for p in model.parameters():
        p.requires_grad = False

    # 收集待替换模块
    to_inject = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(
            name.endswith(sfx) for sfx in target_modules
        ):
            to_inject.append((name, module))

    if not to_inject:
        raise ValueError(
            f"TopLoRA: no Linear matched. target_modules={target_modules}"
        )

    # 执行替换
    total_extra_params = 0
    for name, base_linear in to_inject:
        wrapped = TopLoRALinear(
            base_linear,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        ).to(base_linear.weight.device)
        total_extra_params += wrapped.lora_lambda.weight.numel()
        _set_module_by_path(model, name, wrapped)

    print(f"[TopLoRA] 已注入 {len(to_inject)} 个线性层 (r={r}, alpha={lora_alpha}, dropout={lora_dropout})")
    print(f"[TopLoRA] 额外参数量 (W_lambda): {total_extra_params}")

    # 解冻分类头（与 SoRA inject 保持一致）
    for name, param in model.named_parameters():
        lname = name.lower()
        if (
            "classifier" in lname
            or "score" in lname
            or "lm_head" in lname
            or lname == "shared"
        ):
            param.requires_grad = True
