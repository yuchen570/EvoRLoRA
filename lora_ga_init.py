"""PEFT LoRA 的 LoRA-GA 风格初始化（梯度 SVD）。参见 https://arxiv.org/abs/2407.05000"""
from __future__ import annotations
import copy
import math
from typing import Dict, Iterator, List, Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

def _collect_target_linears(model, target_modules):
    out = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(sfx) for sfx in target_modules):
            out.append((name, module))
    return out

def _forward_loss_nlu(model, batch, loss_fn):
    labels = batch["labels"]
    feats = {k: v for k, v in batch.items() if k != "labels"}
    logits = model(**feats).logits
    if isinstance(loss_fn, nn.MSELoss):
        return loss_fn(logits.squeeze(-1), labels.float())
    return loss_fn(logits, labels)

def _forward_loss_nlg(model, batch):
    out = model(**batch)
    if not hasattr(out, "loss") or out.loss is None:
        raise RuntimeError("NLG LoRA-GA needs Seq2SeqLMOutput.loss")
    return out.loss

def _iter_limited_batches(loader, max_batches):
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        yield batch

def estimate_lora_ga_init_tensors(model, data_loader, target_modules, lora_r, lora_ga_batches, task_type, device, loss_fn=None, stable_gamma=None, direction="ArB2r"):
    """
    通过在批次上累加梯度来估计 LoRA-GA 初始化张量。

    参数：
        direction: SVD 方向策略，可选 "ArBr"、"A2rBr"、"ArB2r"、"random"。
                   匹配原始 LoRA-GA 实现（默认："ArB2r"）。
    """
    if lora_ga_batches < 1:
        raise ValueError("lora_ga_batches >= 1")
    targets = _collect_target_linears(model, target_modules)
    if not targets:
        raise ValueError("LoRA-GA: no target Linear matched")
    model = model.to(device)
    if task_type == "nlu":
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        if isinstance(loss_fn, nn.Module):
            loss_fn = loss_fn.to(device)

    for p in model.parameters():
        p.requires_grad = False

    grad_sums = {}
    for full_name, linear in targets:
        linear.weight.requires_grad = True
        grad_sums[full_name] = torch.zeros_like(linear.weight, device=device, dtype=linear.weight.dtype)

    n_used = 0
    for batch in _iter_limited_batches(data_loader, lora_ga_batches):
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        if task_type == "nlu":
            loss = _forward_loss_nlu(model, batch, loss_fn)
        elif task_type == "nlg":
            loss = _forward_loss_nlg(model, batch)
        else:
            raise ValueError(task_type)
        loss.backward()
        # 累加梯度，不进行裁剪（原始 LoRA-GA 在此处不进行裁剪）
        for full_name, linear in targets:
            w = linear.weight
            if w.grad is None:
                raise RuntimeError("LoRA-GA: no grad for " + full_name)
            grad_sums[full_name] += w.grad.detach()
            w.grad = None
        n_used += 1

    if n_used == 0:
        raise RuntimeError("LoRA-GA: empty loader")

    result = {}
    for full_name, linear in targets:
        linear.weight.requires_grad = False
        G = (grad_sums[full_name] / float(n_used)).detach().cpu().float()
        # 使用 svd_lowrank 以匹配原始 LoRA-GA (q = min(4*r, min(shape)), niter=4)
        q_svd = min(4 * lora_r, min(G.shape))
        U, S, V = torch.svd_lowrank(G, q=q_svd, niter=4)
        V = V.T  # V 现在是 (n, q) -> 转置为 (q, n) 以进行行索引

        # 方向选择：完全匹配原始 LoRA-GA layer.py
        if direction == "ArBr":
            B = U[:, 0: 2 * lora_r: 2]
            A = V[1: 2 * lora_r: 2, :]
        elif direction == "A2rBr":
            B = U[:, :lora_r]
            A = V[lora_r: 2 * lora_r, :]
        elif direction == "ArB2r":
            B = U[:, lora_r: 2 * lora_r]
            A = V[:lora_r, :]
        elif direction == "random":
            import random
            random_list = random.sample(range(2 * lora_r), 2 * lora_r)
            indexes_A = random_list[:lora_r]
            indexes_B = random_list[lora_r:]
            B = U[:, indexes_B]
            A = V[indexes_A, :]
        else:
            raise ValueError(f"Unknown direction: {direction}")

        if stable_gamma is not None:
            # stable 缩放：完全匹配原始 LoRA-GA stable 分支
            # B = B * m**0.25 / gamma**0.5,  A = A * m**0.25 / gamma**0.5
            gamma = float(stable_gamma)
            m = G.shape[0]  # out_features（梯度矩阵的行数）
            scale = m ** 0.25 / gamma ** 0.5
            lora_B = (B * scale).to(dtype=linear.weight.dtype)
            lora_A = (A * scale).to(dtype=linear.weight.dtype)
        else:
            # 备选方案：按奇异值的平方根加权
            sqrt_s = torch.sqrt(torch.clamp(S[:lora_r], min=0.0))
            lora_B = (B * sqrt_s.unsqueeze(0)).to(dtype=linear.weight.dtype)
            lora_A = (sqrt_s.unsqueeze(1) * A).to(dtype=linear.weight.dtype)
        result[full_name] = (lora_A.contiguous(), lora_B.contiguous())
    model.zero_grad(set_to_none=True)
    return result

def _normalize_key_for_peft(full_name):
    return full_name[len("module."):] if full_name.startswith("module.") else full_name

def _peft_module_key_from_full_name(full_name):
    for marker in ("base_model.model.", "base_model."):
        if marker in full_name:
            return full_name[full_name.find(marker) + len(marker):]
    return full_name

def apply_lora_ga_init_to_peft(peft_model, init_by_key, target_device):
    applied = 0
    for full_name, module in peft_model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        key = _normalize_key_for_peft(_peft_module_key_from_full_name(full_name))
        if key not in init_by_key:
            continue
        A_cpu, B_cpu = init_by_key[key]
        lora_A, lora_B = module.lora_A, module.lora_B
        if isinstance(lora_A, nn.ModuleDict):
            adapter = "default" if "default" in lora_A else next(iter(lora_A.keys()))
            la, lb = lora_A[adapter], lora_B[adapter]
        else:
            adapter = "default"
            la, lb = lora_A, lora_B
            
        if not hasattr(la, "weight") or not hasattr(lb, "weight"):
            continue
            
        la.weight.data.copy_(A_cpu.to(device=target_device, dtype=la.weight.dtype))
        lb.weight.data.copy_(B_cpu.to(device=target_device, dtype=lb.weight.dtype))
        
        # --- 基础权重补偿（防止表示崩溃的关键步骤） ---
        if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
            base_weight = module.base_layer.weight
            scaling = 1.0
            if hasattr(module, "scaling") and adapter in module.scaling:
                scaling = module.scaling[adapter]
                
            A_f32 = A_cpu.to(device=target_device, dtype=torch.float32)
            B_f32 = B_cpu.to(device=target_device, dtype=torch.float32)
            offset = (B_f32 @ A_f32) * float(scaling)
            
            base_weight.data.sub_(offset.to(dtype=base_weight.dtype))
            
        applied += 1
    if applied == 0:
        raise RuntimeError("LoRA-GA: no PEFT layer updated")

def broadcast_lora_ga_payload(payload, src=0):
    if not dist.is_available() or not dist.is_initialized():
        assert payload is not None
        return {k: (v[0].clone(), v[1].clone()) for k, v in payload.items()}
    object_list = [payload] if dist.get_rank() == src else [None]
    dist.broadcast_object_list(object_list, src=src)
    out = object_list[0]
    assert out is not None
    return {k: (v[0].clone(), v[1].clone()) for k, v in out.items()}

def build_train_loader_no_sampler(train_loader):
    return DataLoader(
        train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False,
        collate_fn=train_loader.collate_fn, num_workers=getattr(train_loader, "num_workers", 0),
        pin_memory=getattr(train_loader, "pin_memory", False), drop_last=False,
    )

def run_lora_ga_init_pipeline(base_model, train_loader, target_modules, lora_r, lora_ga_batches, task_type, device, is_main_process, ddp_enabled, loss_fn=None, stable_gamma=None, direction="ArB2r"):
    if ddp_enabled and dist.is_available() and dist.is_initialized():
        dist.barrier()
    payload = None
    if is_main_process:
        dl = build_train_loader_no_sampler(train_loader)
        est_model = copy.deepcopy(base_model)
        payload = estimate_lora_ga_init_tensors(est_model, dl, target_modules, lora_r, lora_ga_batches, task_type, device, loss_fn, stable_gamma=stable_gamma, direction=direction)
        del est_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = broadcast_lora_ga_payload(payload, src=0)
    if ddp_enabled and dist.is_available() and dist.is_initialized():
        dist.barrier()
    return out
