"""LoRA-GA style init for PEFT LoRA (gradient SVD). See https://arxiv.org/abs/2407.05000"""
from __future__ import annotations
import copy
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

def estimate_lora_ga_init_tensors(model, data_loader, target_modules, lora_r, lora_ga_batches, task_type, device, loss_fn=None):
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
    result = {}
    for full_name, linear in targets:
        for p in model.parameters():
            p.requires_grad = False
        w = linear.weight
        w.requires_grad = True
        grad_sum = torch.zeros_like(w, device=device, dtype=w.dtype)
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
            if w.grad is None:
                raise RuntimeError("LoRA-GA: no grad for " + full_name)
            grad_sum += w.grad.detach()
            n_used += 1
            w.grad = None
        w.requires_grad = False
        if n_used == 0:
            raise RuntimeError("LoRA-GA: empty loader")
        G = (grad_sum / float(n_used)).detach().cpu().float()
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        r = min(lora_r, S.numel())
        if r < 1:
            raise RuntimeError("LoRA-GA: r < 1")
        sqrt_s = torch.sqrt(torch.clamp(S[:r], min=0.0))
        lora_B = (U[:, :r] * sqrt_s.unsqueeze(0)).to(dtype=w.dtype)
        lora_A = (sqrt_s.unsqueeze(1) * Vh[:r, :]).to(dtype=w.dtype)
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
            la, lb = lora_A, lora_B
        if not hasattr(la, "weight") or not hasattr(lb, "weight"):
            continue
        la.weight.data.copy_(A_cpu.to(device=target_device, dtype=la.weight.dtype))
        lb.weight.data.copy_(B_cpu.to(device=target_device, dtype=lb.weight.dtype))
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

def run_lora_ga_init_pipeline(base_model, train_loader, target_modules, lora_r, lora_ga_batches, task_type, device, is_main_process, ddp_enabled, loss_fn=None):
    if ddp_enabled and dist.is_available() and dist.is_initialized():
        dist.barrier()
    payload = None
    if is_main_process:
        dl = build_train_loader_no_sampler(train_loader)
        est_model = copy.deepcopy(base_model)
        payload = estimate_lora_ga_init_tensors(est_model, dl, target_modules, lora_r, lora_ga_batches, task_type, device, loss_fn)
        del est_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = broadcast_lora_ga_payload(payload, src=0)
    if ddp_enabled and dist.is_available() and dist.is_initialized():
        dist.barrier()
    return out
