"""PEFT LoRA 的 LoRA-GA 风格初始化（梯度 SVD）。参见 https://arxiv.org/abs/2407.05000"""
from __future__ import annotations
import copy
import math
from typing import Dict, Iterator, List, Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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

def estimate_lora_ga_init_tensors(
    model,
    data_loader,
    target_modules,
    lora_r,
    lora_ga_batches,
    task_type,
    device,
    loss_fn=None,
    stable_gamma=None,
    direction="ArB2r",
    aggregate_across_ranks: bool = False,
    svd_on_this_rank: bool = True,
):
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
            # [诊断] 如果首个 batch 的 loss 过高（如 > 5.0），且分类头是随机初始化的，
            # 意味着梯度估计是在“瞎猜”，此时生成的 LoRA 矩阵质量会很差。
            if n_used == 0 and loss.item() > 5.0:
                print(f"[warning] LoRA-GA: 初始 loss={loss.item():.4f} 异常高。如果分类头(classifier/score)是随机初始化的，梯度估计可能不准。")
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

    # 对齐官方 LoRA-GA 多进程路径：各 rank 本地估计后做 all_reduce 聚合。
    if aggregate_across_ranks and dist.is_available() and dist.is_initialized():
        n_used_t = torch.tensor(float(n_used), device=device, dtype=torch.float32)
        dist.all_reduce(n_used_t, op=dist.ReduceOp.SUM)
        n_used_global = float(n_used_t.item())
        if n_used_global <= 0:
            raise RuntimeError("LoRA-GA: global n_used == 0 after all_reduce")
        for full_name in grad_sums:
            dist.all_reduce(grad_sums[full_name], op=dist.ReduceOp.SUM)
        n_used = n_used_global

    if not svd_on_this_rank:
        # 非主 rank 不做 SVD，等待主 rank 广播初始化结果。
        model.zero_grad(set_to_none=True)
        return None

    result = {}
    for full_name, linear in targets:
        linear.weight.requires_grad = False
        # 计算平均梯度
        G = (grad_sums[full_name] / float(n_used)).detach().cpu().float()
        
        # 使用 svd_lowrank 以匹配原始 LoRA-GA (q = min(4*r, min(shape)), niter=4)
        q_svd = min(4 * lora_r, min(G.shape))
        try:
            U, S, V = torch.svd_lowrank(G, q=q_svd, niter=4)
            V = V.T  # V 现在是 (q, n)
        except Exception as e:
            print(f"[error] LoRA-GA SVD failed for {full_name}: {e}")
            # 回退到随机初始化或全 0
            B = torch.zeros((G.shape[0], lora_r))
            A = torch.zeros((lora_r, G.shape[1]))
            result[full_name] = (A, B)
            continue

        # 方向选择策略：匹配原始 LoRA-GA layer.py
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
            # Stable 缩放策略：B = B * m**0.25 / gamma**0.5, A = A * m**0.25 / gamma**0.5
            gamma = float(stable_gamma)
            m = G.shape[0]  # out_features
            scale = (m ** 0.25) / (gamma ** 0.5)
            lora_B = B * scale
            lora_A = A * scale
        else:
            # 默认：按奇异值平方根缩放
            sqrt_s = torch.sqrt(torch.clamp(S[:lora_r], min=0.0))
            lora_B = B * sqrt_s.unsqueeze(0)
            lora_A = sqrt_s.unsqueeze(1) * A
            
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
        
        # --- 基础权重补偿（核心步骤：W_new = W_pretrained - scaling * (B @ A)） ---
        # 这样在推理时，W_new + scaling * (B @ A) 刚好等于 W_pretrained，实现 Step 0 无损初始化。
        if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
            base_weight = module.base_layer.weight
            scaling = 1.0
            if hasattr(module, "scaling") and adapter in module.scaling:
                scaling = module.scaling[adapter]

            # --- 计算初始范数和最大值，用于后续裁剪判断 ---
            w_abs_max = base_weight.data.float().abs().max()
            o_abs_max = offset.abs().max()
            w_norm = base_weight.data.float().norm().item()
            o_norm = offset.norm().item()
            ratio_norm = o_norm / max(w_norm, 1e-12)

            # --- 第一重保护：10% 范数裁剪 (Phase 6/9) ---
            # 如果补偿量 (offset) 超过原始权重的 10%，说明估计梯度干扰过大，需按比例压制。
            MAX_RATIO = 0.1
            if ratio_norm > MAX_RATIO:
                scale_ratio = MAX_RATIO / ratio_norm
                print(f"[warning] LoRA-GA layer={key}: ratio={ratio_norm:.2%} > {MAX_RATIO:.1%}, scaling down by {scale_ratio:.4f}")
                offset *= scale_ratio
                # 必须同步缩放 A 和 B 分量，否则 A@B 会失去比例
                sqrt_scale = math.sqrt(scale_ratio)
                la.weight.data.mul_(sqrt_scale)
                lb.weight.data.mul_(sqrt_scale)
                
                # 重要：缩放后需重新计算 o_norm 和 o_abs_max，供下一重保护使用
                o_norm = offset.norm().item()
                o_abs_max = offset.abs().max()
                ratio_norm = o_norm / max(w_norm, 1e-12)

            # --- 第二重保护：abs_max 阈值裁剪 (对齐 peft 官方/原始实现) ---
            # 确保 offset 的最大元素不超过权重矩阵的最大元素，防止局部权重被削成负数后发散。
            if o_abs_max > 0 and w_abs_max / o_abs_max < 1.0:
                ratio = (w_abs_max / o_abs_max).item()
                print(f"[info] LoRA-GA: layer={key} clipping offset by ratio={ratio:.4f} (abs_max guard)")
                offset *= ratio
                sqrt_ratio = math.sqrt(ratio)
                la.weight.data.mul_(sqrt_ratio)
                lb.weight.data.mul_(sqrt_ratio)

            # --- 执行补偿：W_new = W_pretrained - offset ---
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

def build_lora_ga_estimation_loader(train_loader, ddp_enabled: bool):
    if ddp_enabled and dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            train_loader.dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False,
        )
        return DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            sampler=sampler,
            collate_fn=train_loader.collate_fn,
            num_workers=getattr(train_loader, "num_workers", 0),
            pin_memory=getattr(train_loader, "pin_memory", False),
            drop_last=False,
        )
    return DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        collate_fn=train_loader.collate_fn,
        num_workers=getattr(train_loader, "num_workers", 0),
        pin_memory=getattr(train_loader, "pin_memory", False),
        drop_last=False,
    )

def run_lora_ga_init_pipeline(base_model, train_loader, target_modules, lora_r, lora_ga_batches, task_type, device, is_main_process, ddp_enabled, loss_fn=None, stable_gamma=None, direction="ArB2r"):
    # --- 只在计算主 Rank 执行初始化估计 ---
    # 虽然各 Rank 都有聚合梯度，但为了绝对保证位级别一致性，我们强制 Rank 0 做 SVD 并广播。
    payload = None
    if is_main_process:
        dl = build_lora_ga_estimation_loader(train_loader, ddp_enabled=ddp_enabled)
        est_model = copy.deepcopy(base_model)
        payload = estimate_lora_ga_init_tensors(
            est_model,
            dl,
            target_modules,
            lora_r,
            lora_ga_batches,
            task_type,
            device,
            loss_fn,
            stable_gamma=stable_gamma,
            direction=direction,
            aggregate_across_ranks=bool(ddp_enabled and dist.is_available() and dist.is_initialized()),
            svd_on_this_rank=True,
        )
        del est_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 将 Rank 0 计算的结果广播到所有节点
    payload = broadcast_lora_ga_payload(payload, src=0)
    
    if ddp_enabled and dist.is_available() and dist.is_initialized():
        dist.barrier()
    return payload
