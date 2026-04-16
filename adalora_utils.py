"""
AdaLoRA 训练辅助：正交正则与 update_and_allocate 解析。

本仓库用 DictFeatureClassifier 只取 logits、在外部算 CE/seq2seq loss，不会触发 PEFT AdaLoraModel.forward
里对 outputs.loss 注入的正交项，因此需在此处显式加入（与 PEFT 实现一致，见 peft AdaLoraModel.forward）。
论文：AdaLoRA (Zhang et al., ICLR 2023) https://arxiv.org/abs/2303.10512
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple


def compute_adalora_orthogonal_loss(inner: nn.Module, adapter_name: str = "default") -> torch.Tensor:
    """
    对含 adapter_name 的 lora_A / lora_B 计算 Frobenius||P P^T - I||（A 用 P@P^T，B 用 P^T@P），再对矩阵数取平均。
    返回与 inner 同 device/dtype 的标量（未乘 orth_reg_weight）。
    """
    device = next(inner.parameters()).device
    dtype = next(inner.parameters()).dtype
    regu = torch.zeros((), device=device, dtype=dtype)
    num_param = 0
    for name, p in inner.named_parameters():
        if adapter_name not in name:
            continue
        if "lora_A" not in name and "lora_B" not in name:
            continue
        if p.numel() == 0:
            continue
        para_cov = p @ p.T if "lora_A" in name else p.T @ p
        eye = torch.eye(para_cov.size(0), device=para_cov.device, dtype=para_cov.dtype, requires_grad=False)
        regu = regu + torch.norm(para_cov - eye, p="fro")
        num_param += 1
    if num_param == 0:
        return regu
    return regu / float(num_param)


def get_adalora_orth_reg_weight(inner: nn.Module, adapter_name: str = "default") -> float:
    cfg = getattr(inner, "peft_config", None)
    if cfg is None or adapter_name not in cfg:
        return 0.0
    w = getattr(cfg[adapter_name], "orth_reg_weight", 0.0)
    return float(w)


def unwrap_inner_from_training_model(model: nn.Module) -> nn.Module:
    """DDP(DictFeatureClassifier(inner)) -> inner (PeftModel)。"""
    m = model.module if hasattr(model, "module") else model
    return getattr(m, "inner", m)


def adalora_update_and_allocate(model: nn.Module, global_step: int) -> None:
    """
    调用 PEFT AdaLora 的预算更新；兼容 DDP + DictFeatureClassifier 包装。
    """
    inner = unwrap_inner_from_training_model(model)
    candidates = [
        getattr(inner, "base_model", None),
        inner,
        getattr(inner, "model", None),
    ]
    for c in candidates:
        if c is not None and hasattr(c, "update_and_allocate"):
            c.update_and_allocate(global_step)
            return
    for mod in inner.modules():
        if mod is not inner and hasattr(mod, "update_and_allocate"):
            mod.update_and_allocate(global_step)
            return


def normalize_adalora_schedule(
    total_steps: int,
    adalora_tinit: Optional[int] = None,
    adalora_tfinal: Optional[int] = None,
) -> Tuple[int, int, Optional[str]]:
    """
    规范化 AdaLoRA 调度，保证 tinit < total_step - tfinal（中间段至少 1 步）。
    返回: (tinit, tfinal, warn_message_or_none)
    """
    planned_steps = max(int(total_steps or 1000), 1)

    if adalora_tinit is None:
        tinit = max(int(0.1 * planned_steps), 1)
    else:
        tinit = max(int(adalora_tinit), 1)

    if adalora_tfinal is None:
        tfinal = max(int(0.1 * planned_steps), tinit + 1)
    else:
        tfinal = max(int(adalora_tfinal), tinit + 1)

    warn_message: Optional[str] = None
    if tinit >= planned_steps - tfinal:
        orig_tinit, orig_tfinal = tinit, tfinal
        cap = planned_steps - 2  # 严格保留 tinit、tfinal 与至少 1 步中间段
        if cap < 2:
            raise ValueError(
                f"AdaLoRA 调度无效：total_step={planned_steps} 过短，无法满足 tinit/tfinal。"
                f"当前 tinit={tinit}, tfinal={tfinal}。请增大训练步数或减小 adalora_tinit/adalora_tfinal。"
            )

        sum_ab = tinit + tfinal
        scale = float(cap) / float(max(sum_ab, 1))
        tinit = max(1, int(tinit * scale))
        tfinal = max(tinit + 1, int(tfinal * scale))
        if tinit + tfinal > cap:
            tfinal = cap - tinit
        tfinal = max(tfinal, tinit + 1)

        if tinit + tfinal >= planned_steps:
            tinit = max(1, planned_steps // 3)
            tfinal = max(tinit + 1, min(orig_tfinal, planned_steps - tinit - 1))

        if tinit >= planned_steps - tfinal:
            raise ValueError(
                f"AdaLoRA 调度无效：需满足 tinit < total_step - tfinal，当前 total_step={planned_steps}, "
                f"tinit={tinit}, tfinal={tfinal}（已尝试按步数收紧仍失败）。请增大训练步数或显式调小 adalora_tinit/adalora_tfinal。"
            )

        warn_message = (
            f"[adalora] tinit/tfinal 与 total_step={planned_steps} 不兼容（官方值常针对更长训练），"
            f"已按比例收紧: tinit {orig_tinit}->{tinit}, tfinal {orig_tfinal}->{tfinal}。"
        )

    return tinit, tfinal, warn_message
