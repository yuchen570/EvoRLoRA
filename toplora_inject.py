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
    - lambda_clamp: exp 输入 clamp ±C（数值稳定化）。官方实现未 clamp，在高 lr 长训练
      下观测到 seed 级崩溃（某层 RMSNorm 输出峰值→exp 爆炸→分类头坍缩为常数输出）。
      设 0/负数可关闭（等价于官方原始实现）。
    """

    def __init__(self, token_dim: int, r: int, dtype=None, lambda_clamp: float = 3.0):
        super().__init__()
        self.r = r
        self.token_dim = token_dim
        self.lambda_clamp = float(lambda_clamp)
        self.weight = nn.Parameter(torch.empty((token_dim, r), dtype=dtype))
        self.rms_norm = nn.RMSNorm([r])
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), mode="fan_out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = x @ self.weight
        weight = self.rms_norm(weight)
        if self.lambda_clamp > 0:
            # λ ∈ [e^-C, e^C]，默认 C=3.0 → λ ∈ [0.05, 20.09]，足以保留 token-wise 缩放表达力
            weight = weight.clamp(min=-self.lambda_clamp, max=self.lambda_clamp)
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
        lambda_clamp: float = 3.0,
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
        self.lora_lambda = TopSingularValue(token_dim=d_in, r=r, lambda_clamp=lambda_clamp)

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
    lambda_clamp: float = 3.0,
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
    for name, base_linear in to_inject:
        wrapped = TopLoRALinear(
            base_linear,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lambda_clamp=lambda_clamp,
        ).to(base_linear.weight.device)
        _set_module_by_path(model, name, wrapped)

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
