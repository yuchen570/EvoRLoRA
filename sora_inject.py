"""SoRA 风格的稀疏门控 LoRA 分支。"""
from __future__ import annotations
import math
from typing import List
import numpy as np
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
        lname = name.lower()
        if (
            "classifier" in lname
            or "score" in lname
            or "lm_head" in lname
            or lname == "shared"
        ):
            param.requires_grad = True
        # pooler 和 LayerNorm 不再默认解冻，由外部调用方按需控制（公平对比要求与其他方法一致）


class SparseAdamW(torch.optim.AdamW):
    """
    近端梯度 (Proximal Gradient) 的 AdamW，用于 SoRA gate 参数。

    完全复现原始 SoRA src/sparse_optimizer.py 的行为：
    - 继承 transformers.AdamW 风格，支持 correct_bias 参数（bias correction 开关）
    - 自己实现完整的 AdamW 更新循环（不依赖 super().step()），与原始实现结构一致
    - 软阈值只对有梯度的参数应用（p.grad is None 时跳过）
    - 阈值直接使用 sparse_lambda（不乘以 lr），与 sparse_optimizer.py 一致

    参考：TsinghuaC3I/SoRA (EMNLP 2023) src/sparse_optimizer.py
    """

    def __init__(
        self,
        params,
        sparse_lambda: float = 1e-3,
        correct_bias: bool = True,
        lambda_schedule: str | None = None,
        max_lambda: float | None = None,
        lambda_num: int | None = None,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self.sparse_lambda = sparse_lambda
        self.correct_bias = correct_bias
        self.lambda_schedule = lambda_schedule
        self.lambda_idx = 0
        self._build_lambda_list(max_lambda=max_lambda, lambda_num=lambda_num)

    def _build_lambda_list(self, max_lambda: float | None, lambda_num: int | None) -> None:
        if self.lambda_schedule is None:
            self._lambdas = None
            return
        if isinstance(self.lambda_schedule, list):
            self._lambdas = list(self.lambda_schedule)
            return
        if max_lambda is None or lambda_num is None:
            raise ValueError("SoRA lambda schedule 需要同时提供 max_lambda 和 lambda_num")
        if self.lambda_schedule == "linear":
            self._lambdas = np.linspace(self.sparse_lambda, float(max_lambda), int(lambda_num)).tolist()
        elif self.lambda_schedule == "log_linear":
            self._lambdas = np.log(
                np.linspace(np.exp(self.sparse_lambda), np.exp(float(max_lambda)), int(lambda_num))
            ).tolist()
        elif self.lambda_schedule == "exp_linear":
            self._lambdas = np.exp(
                np.linspace(np.log(self.sparse_lambda), np.log(float(max_lambda)), int(lambda_num))
            ).tolist()
        else:
            raise NotImplementedError(f"未知 SoRA lambda_schedule: {self.lambda_schedule}")

    def step_lambda(self) -> None:
        if not self._lambdas:
            return
        if self.lambda_idx < len(self._lambdas) - 1:
            self.lambda_idx += 1
            self.sparse_lambda = float(self._lambdas[self.lambda_idx])

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                # correct_bias: bias correction，对应 transformers.AdamW 的 correct_bias 参数
                if self.correct_bias:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # gate 参数不需要 weight decay（原始注释："params with sparsity regularization do not need weight decay"）
                to_add = torch.div(exp_avg, denom) * (-step_size)
                if group["weight_decay"] > 0.0:
                    to_add = to_add + (-group["lr"] * group["weight_decay"]) * p.data
                p.data.add_(to_add)

                # Soft-thresholding: θ ← sign(θ) · max(|θ| - λ, 0)
                # 阈值直接使用 sparse_lambda（不乘以 lr），与原始 sparse_optimizer.py 一致
                if self.sparse_lambda > 0:
                    p.data = torch.sign(p.data) * torch.clamp(p.data.abs() - self.sparse_lambda, min=0.0)

        return loss
