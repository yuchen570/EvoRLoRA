import argparse
import copy
import csv
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
)

from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model

from rank_evolution_controller import RankEvolutionController
from train_integration import inject_evo_lora, train_evo_lora_step

from torch.nn.parallel import DistributedDataParallel as DDP


class DictFeatureClassifier(nn.Module):
    """
    适配 train_evo_lora_step 的输入约定：
    - 输入是特征字典（input_ids/attention_mask/...）
    - 输出是分类 logits
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.inner(**features).logits


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def extract_features_and_labels(
    batch: Dict[str, torch.Tensor],
    task_type: str = "nlu",
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    - NLU：features 去掉 labels，targets 作为 labels 单独返回。
    - NLG：为了让 encoder-decoder 模型在 forward 时可基于 labels 推导 decoder 输入，
      features 保留 labels；同时 targets 仍单独返回用于计算 token-level loss。
    """
    labels = batch["labels"]
    if task_type == "nlg":
        return dict(batch), labels
    features = {k: v for k, v in batch.items() if k != "labels"}
    return features, labels


def setup_data_and_model(
    task_name: str = "sst2",
    model_name: str = "roberta-base",
    batch_size: int = 16,
    max_length: int = 128,
    task_type: str = "nlu",
    nlg_dataset_name: str = "cnn_dailymail",
    max_target_length: int = 64,
    dataset_cache_dir: str = "datasets",
    model_cache_dir: str = "models",
    ddp_enabled: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, nn.Module, AutoTokenizer]:
    os.makedirs(dataset_cache_dir, exist_ok=True)
    os.makedirs(model_cache_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)

    if task_type == "nlu":
        dataset = load_dataset("glue", task_name, cache_dir=dataset_cache_dir)

        sentence_keys = {
            "sst2": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "qnli": ("question", "sentence"),
            "qqp": ("question1", "question2"),
            "rte": ("sentence1", "sentence2"),
        }
        if task_name not in sentence_keys:
            raise ValueError(
                f"当前脚本只内置了 {list(sentence_keys.keys())} 的字段映射，请扩展后再用: {task_name}"
            )

        sentence1_key, sentence2_key = sentence_keys[task_name]

        def tokenize_fn(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
            if sentence2_key is None:
                return tokenizer(examples[sentence1_key], truncation=True, max_length=max_length)
            return tokenizer(
                examples[sentence1_key],
                examples[sentence2_key],
                truncation=True,
                max_length=max_length,
            )

        tokenized = dataset.map(tokenize_fn, batched=True)
        tokenized = tokenized.rename_column("label", "labels")
        keep_cols = ["input_ids", "attention_mask", "labels"]
        if "token_type_ids" in tokenized["train"].column_names:
            keep_cols.append("token_type_ids")
        tokenized = tokenized.remove_columns([c for c in tokenized["train"].column_names if c not in keep_cols])
        tokenized.set_format(type="torch")

        collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        val_split_name = "validation_matched" if task_name == "mnli" else "validation"

        if ddp_enabled and world_size > 1:
            train_sampler = DistributedSampler(
                tokenized["train"],
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
                drop_last=False,
            )
            # 为了让 mini_val_k 的批次数在各卡上尽量一致，这里使用 drop_last=True。
            val_sampler = DistributedSampler(
                tokenized[val_split_name],
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=seed,
                drop_last=True,
            )
            train_loader = DataLoader(
                tokenized["train"], batch_size=batch_size, sampler=train_sampler, collate_fn=collator
            )
            val_loader = DataLoader(
                tokenized[val_split_name], batch_size=batch_size, sampler=val_sampler, collate_fn=collator
            )
        else:
            train_loader = DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True, collate_fn=collator)
            val_loader = DataLoader(tokenized[val_split_name], batch_size=batch_size, shuffle=False, collate_fn=collator)

        num_labels = 2 if task_name == "sst2" else None
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            cache_dir=model_cache_dir,
        )
        return train_loader, val_loader, base_model, tokenizer

    if task_type == "nlg":
        if nlg_dataset_name != "cnn_dailymail":
            raise NotImplementedError("当前仅支持 nlg_dataset_name='cnn_dailymail'")

        dataset = load_dataset(nlg_dataset_name, "3.0.0", cache_dir=dataset_cache_dir)
        text_key = "article"
        target_key = "highlights"

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        def preprocess(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
            inputs = examples[text_key]
            targets = examples[target_key]
            model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized = dataset.map(
            preprocess,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        tokenized.set_format(type="torch")

        # seq2seq collator：会把 labels 的 pad 位置替换为 -100（用于 ignore_index）
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)
        val_split_name = "validation"

        if ddp_enabled and world_size > 1:
            train_sampler = DistributedSampler(
                tokenized["train"],
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
                drop_last=False,
            )
            val_sampler = DistributedSampler(
                tokenized[val_split_name],
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=seed,
                drop_last=True,
            )
            train_loader = DataLoader(
                tokenized["train"], batch_size=batch_size, sampler=train_sampler, collate_fn=collator
            )
            val_loader = DataLoader(
                tokenized[val_split_name], batch_size=batch_size, sampler=val_sampler, collate_fn=collator
            )
        else:
            train_loader = DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True, collate_fn=collator)
            val_loader = DataLoader(tokenized[val_split_name], batch_size=batch_size, shuffle=False, collate_fn=collator)

        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=model_cache_dir)
        return train_loader, val_loader, base_model, tokenizer

    raise ValueError(f"未知 task_type: {task_type}")


def peft_factory(
    model: nn.Module,
    method_name: str,
    target_rank: int = 8,
    total_steps: Optional[int] = None,
    adalora_delta_t: int = 200,
) -> Tuple[nn.Module, Optional[RankEvolutionController], Dict[str, Any]]:
    model_type = getattr(getattr(model, "config", None), "model_type", "").lower()
    if "roberta" in model_type or "bert" in model_type:
        target_modules = ["query", "value"]
    elif "llama" in model_type or "mistral" in model_type:
        target_modules = ["q_proj", "v_proj"]
    elif "t5" in model_type:
        # T5Attention 里常见的投影矩阵名为 q / k / v
        target_modules = ["q", "v"]
    else:
        # 回退默认：优先兼容 BERT/RoBERTa 风格命名
        target_modules = ["query", "value"]

    # PEFT 的 task_type 需要与 backbone 类型匹配，避免 seq2seq/causal 分支内部逻辑错误。
    if "t5" in model_type:
        peft_task_type = TaskType.SEQ_2_SEQ_LM
    elif "llama" in model_type or "mistral" in model_type:
        peft_task_type = TaskType.CAUSAL_LM
    else:
        peft_task_type = TaskType.SEQ_CLS

    controller: Optional[RankEvolutionController] = None

    if method_name == "lora":
        config = LoraConfig(
            task_type=peft_task_type,
            r=target_rank,
            lora_alpha=2 * target_rank,
            lora_dropout=0.1,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, config)

    elif method_name == "adalora":
        # AdaLoRA 时间超参与训练总步数对齐，避免动态预算未触发或过早结束。
        planned_steps = max(int(total_steps or 1000), 1)
        tinit = max(int(0.1 * planned_steps), 1)
        tfinal = max(int(0.8 * planned_steps), tinit + 1)
        config = AdaLoraConfig(
            task_type=peft_task_type,
            init_r=target_rank * 2,
            target_r=target_rank,
            lora_alpha=2 * target_rank,
            lora_dropout=0.1,
            target_modules=target_modules,
            beta1=0.85,
            beta2=0.85,
            total_step=planned_steps,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=adalora_delta_t,
        )
        model = get_peft_model(model, config)

    elif method_name == "evorank":
        controller = inject_evo_lora(
            model=model,
            target_modules=target_modules,
            layer_kwargs={"r_max": 16, "r_init": target_rank, "lora_alpha": 2.0 * target_rank},
            controller_kwargs={"rho": 0.9, "p_g": 0.8, "p_p": 0.1, "H_g": 2, "H_p": 3, "cooldown_steps": 2},
        )
        # EvoRank 手动注入后，需要显式解冻任务头（与 HF PEFT 在 SEQ_CLS 下的行为对齐）。
        # NLU 通常是 classifier.*（或 score.*）；NLG Seq2Seq 通常为 lm_head/shared。
        for name, param in model.named_parameters():
            if "classifier" in name or "score" in name or "lm_head" in name or name == "shared":
                param.requires_grad = True

    elif method_name in {"lora-ga", "sora", "mtl-lora"}:
        raise NotImplementedError(
            f"{method_name} 尚未接入。\n"
            "TODO: 从对应官方仓库导入构造函数，并保证统一接口 (model, target_rank) -> peft_model。"
        )
    else:
        raise ValueError(f"未知 method_name: {method_name}")

    wrapped_model = DictFeatureClassifier(model)
    trainable_params = count_trainable_params(wrapped_model)
    meta = {"trainable_params": trainable_params}
    return wrapped_model, controller, meta


def _adalora_post_step_update(model: nn.Module, global_step: int) -> None:
    """
    兼容不同 PEFT 版本下 AdaLoRA 的步后预算更新入口。
    """
    if hasattr(model, "update_and_allocate"):
        model.update_and_allocate(global_step)
        return
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "update_and_allocate"):
        base_model.update_and_allocate(global_step)


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    method_name: str,
    controller: Optional[RankEvolutionController] = None,
    task_type: str = "nlu",
    tokenizer: Optional[AutoTokenizer] = None,
    generation_max_new_tokens: int = 64,
    nlg_eval_max_samples: int = 200,
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    T_es: int = 200,
    mini_val_k: int = 8,
    max_train_steps: Optional[int] = None,
    log_dir: str = "runs/benchmark",
    use_wandb: bool = False,
    wandb_project: str = "evorank-benchmark",
    is_main_process: bool = True,
    ddp_enabled: bool = False,
    local_rank: int = 0,
    lambda_c: float = 0.0,
    complexity_mode: str = "rank_sum",
    lambda_pop: Optional[int] = None,
    population_strategy: str = "all",
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    if ddp_enabled and dist.is_available() and dist.is_initialized():
        if not torch.cuda.is_available():
            raise RuntimeError("DDP 目前仅支持 CUDA，但未检测到 CUDA")
        # 关键：每个进程必须绑定到自己的 local GPU，避免 NCCL Duplicate GPU
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 在同进程串行跑多方法时，先清空缓存可显著降低碎片化导致的 OOM 风险。
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device)
    if ddp_enabled and dist.is_available() and dist.is_initialized():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    if task_type == "nlu":
        loss_fn = nn.CrossEntropyLoss()
    elif task_type == "nlg":
        # logits: (B, T, V), labels: (B, T) with padding masked as -100
        def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            vocab = logits.size(-1)
            return F.cross_entropy(
                logits.view(-1, vocab),
                labels.view(-1),
                ignore_index=-100,
            )

    else:
        raise ValueError(f"未知 task_type: {task_type}")

    total_train_steps = max_train_steps if max_train_steps is not None else epochs * len(train_loader)
    warmup_steps = int(total_train_steps * warmup_ratio)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )
    writer = SummaryWriter(log_dir=os.path.join(log_dir, method_name)) if is_main_process else None

    wandb_run = None
    if is_main_process and use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(project=wandb_project, name=method_name, config={"method": method_name})
        except Exception:
            wandb_run = None

    # mini-val 缓存采用 CPU 安全策略：Trial 时按需搬运到模型 device。
    mini_val_batches: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]] = []
    for i, vb in enumerate(val_loader):
        if i >= mini_val_k:
            break
        feats, labels = extract_features_and_labels(vb, task_type=task_type)
        # 确保缓存为 CPU（避免占用显存且降低缓存滞留风险）
        feats = {k: v.detach().cpu() for k, v in feats.items()}
        labels = labels.detach().cpu()
        mini_val_batches.append((feats, labels))

    start_time = time.perf_counter()
    global_step = 0
    best_val_acc = 0.0
    train_loss_ema = None
    ema_beta = 0.95

    for epoch in range(epochs):
        model.train()
        # DistributedSampler 在每个 epoch 都要 set_epoch，否则会导致采样不同步。
        if ddp_enabled and hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            features, labels = extract_features_and_labels(batch, task_type=task_type)

            if method_name == "evorank":
                if controller is None:
                    raise ValueError("evorank 方法必须传入 controller")
                out = train_evo_lora_step(
                    model=model,
                    controller=controller,
                    optimizer=optimizer,
                    train_batch=(features, labels),
                    val_batch=mini_val_batches,
                    loss_fn=loss_fn,
                    step=global_step,
                    warmup_steps=warmup_steps,
                    T_es=T_es,
                    lambda_c=lambda_c,
                    complexity_mode=complexity_mode,
                    lambda_pop=lambda_pop,
                    population_strategy=population_strategy,
                    random_seed=random_seed,
                )
                train_loss = float(out["train_loss"])
                avg_active_rank = float(
                    sum(layer.get_active_rank() for layer in controller.layers.values()) / max(len(controller.layers), 1)
                )
            else:
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                if method_name == "adalora":
                    _adalora_post_step_update(model, global_step)
                train_loss = float(loss.detach().item())
                avg_active_rank = float("nan")

            # 无论哪条路径，optimizer.step() 都已在本步完成，此处统一推进学习率调度。
            lr_scheduler.step()
            current_lr = float(optimizer.param_groups[0]["lr"])

            train_loss_ema = train_loss if train_loss_ema is None else (ema_beta * train_loss_ema + (1 - ema_beta) * train_loss)
            if writer is not None:
                writer.add_scalar("train/loss", train_loss, global_step)
                writer.add_scalar("train/loss_ema", train_loss_ema, global_step)
                writer.add_scalar("train/lr", current_lr, global_step)
                if method_name == "evorank":
                    writer.add_scalar("train/active_rank_mean", avg_active_rank, global_step)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": train_loss,
                        "train/loss_ema": train_loss_ema,
                        "train/lr": current_lr,
                        "step": global_step,
                    }
                )

            global_step += 1
            if max_train_steps is not None and global_step >= max_train_steps:
                break

        # 每个 epoch 在完整验证集上评估指标
        model.eval()

        if task_type == "nlu":
            correct = 0
            total = 0
            with torch.no_grad():
                for vb in val_loader:
                    vb = batch_to_device(vb, device)
                    features, labels = extract_features_and_labels(vb, task_type=task_type)
                    logits = model(features)
                    preds = torch.argmax(logits, dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.numel()
            val_metric = correct / max(total, 1)
            # DDP 下统计全局 accuracy：把 correct/total 做 all_reduce。
            if ddp_enabled and dist.is_available() and dist.is_initialized():
                stats = torch.tensor([correct, total], device=device, dtype=torch.long)
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                correct = int(stats[0].item())
                total = int(stats[1].item())
                val_metric = correct / max(total, 1)
            best_val_acc = max(best_val_acc, val_metric)

            if is_main_process:
                print(
                    f"[{method_name}] epoch={epoch + 1}/{epochs} "
                    f"step={global_step} val_acc={val_metric:.4f} best={best_val_acc:.4f}"
                )
                if writer is not None:
                    writer.add_scalar("val/accuracy", val_metric, epoch)
                if wandb_run is not None:
                    wandb_run.log({"val/accuracy": val_metric, "epoch": epoch + 1, "step": global_step})

        else:
            if tokenizer is None:
                raise ValueError("task_type='nlg' 时必须传入 tokenizer")
            if not is_main_process:
                # 非主进程只跑训练，不做生成评估
                val_metric = 0.0
            else:
                try:
                    import evaluate  # type: ignore

                    rouge_metric = evaluate.load("rouge")
                except Exception:
                    rouge_metric = None

                # 生成阶段需要底层 seq2seq 模型（可能在 DictFeatureClassifier.inner 或 DDP.module.inner 中）
                if isinstance(model, DDP):
                    inner = getattr(model.module, "inner", model.module)
                    gen_model = inner
                else:
                    gen_model = getattr(model, "inner", model)

                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

                preds_text: List[str] = []
                refs_text: List[str] = []
                sample_count = 0

                with torch.no_grad():
                    for vb in val_loader:
                        vb = batch_to_device(vb, device)
                        input_ids = vb["input_ids"]
                        attention_mask = vb.get("attention_mask", None)
                        labels = vb["labels"]

                        gen_kwargs = dict(input_ids=input_ids, max_new_tokens=generation_max_new_tokens)
                        if attention_mask is not None:
                            gen_kwargs["attention_mask"] = attention_mask
                        gen_ids = gen_model.generate(**gen_kwargs)
                        gen_ids_cpu = gen_ids.detach().cpu()
                        pred_batch = tokenizer.batch_decode(gen_ids_cpu, skip_special_tokens=True)

                        # labels 里 padding 会是 -100，需要还原为 pad_id 便于 decode
                        labels_ids = labels.clone()
                        labels_ids[labels_ids == -100] = pad_id
                        labels_ids_cpu = labels_ids.detach().cpu()
                        ref_batch = tokenizer.batch_decode(labels_ids_cpu, skip_special_tokens=True)

                        preds_text.extend(pred_batch)
                        refs_text.extend(ref_batch)
                        sample_count += len(pred_batch)
                        if sample_count >= nlg_eval_max_samples:
                            break

                if rouge_metric is None:
                    val_metric = 0.0
                else:
                    scores = rouge_metric.compute(predictions=preds_text, references=refs_text)
                    # evaluate 的 key 可能是 rougeL / rougeLsum，取存在的那个
                    val_metric = float(scores.get("rougeL", scores.get("rougeLsum", 0.0)))

                best_val_acc = max(best_val_acc, val_metric)
                if is_main_process:
                    print(
                        f"[{method_name}] epoch={epoch + 1}/{epochs} "
                        f"step={global_step} val_rougeL={val_metric:.4f} best={best_val_acc:.4f}"
                    )
                    if writer is not None:
                        writer.add_scalar("val/rougeL", val_metric, epoch)
                    if wandb_run is not None:
                        wandb_run.log({"val/rougeL": val_metric, "epoch": epoch + 1, "step": global_step})

        if max_train_steps is not None and global_step >= max_train_steps:
            break

    total_time = time.perf_counter() - start_time
    peak_mem_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if torch.cuda.is_available()
        else 0.0
    )

    if writer is not None:
        writer.add_scalar("system/peak_memory_mb", peak_mem_mb, 0)
        writer.close()
    if wandb_run is not None:
        wandb_run.finish()

    result = {
        "method": method_name,
        "best_val_accuracy": best_val_acc,
        "total_train_time_sec": total_time,
        "peak_memory_mb": peak_mem_mb,
        "avg_active_rank": (
            float(
                sum(layer.get_active_rank() for layer in controller.layers.values())
                / max(len(controller.layers), 1)
            )
            if method_name == "evorank" and controller is not None
            else "N/A"
        ),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EvoRank-LoRA GLUE benchmark runner")
    parser.add_argument("--task_name", type=str, default="sst2")
    parser.add_argument("--task_type", type=str, default="nlu", choices=["nlu", "nlg"], help="nlu=GLUE分类，nlg=文本生成")
    parser.add_argument("--nlg_dataset_name", type=str, default="cnn_dailymail", help="nlg 数据集名（如 cnn_dailymail）")
    parser.add_argument("--max_target_length", type=int, default=64, help="nlg 的摘要/目标序列最大长度")
    parser.add_argument("--generation_max_new_tokens", type=int, default=64, help="nlg 生成最大新 tokens")
    parser.add_argument("--nlg_eval_max_samples", type=int, default=200, help="nlg 验证集评测最大样本数（仅用于 ROUGE）")
    parser.add_argument("--dataset_cache_dir", type=str, default="datasets", help="数据集缓存目录（建议仓库内相对路径）")
    parser.add_argument("--model_cache_dir", type=str, default="models", help="模型缓存目录（建议仓库内相对路径）")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--methods", nargs="+", default=["lora", "adalora", "evorank"])
    parser.add_argument("--target_rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--T_es", type=int, default=200)
    parser.add_argument("--mini_val_k", type=int, default=8)
    parser.add_argument("--adalora_delta_t", type=int, default=200)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default="runs/benchmark")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="evorank-benchmark")
    parser.add_argument("--lambda_c", type=float, default=0.0)
    parser.add_argument("--complexity_mode", type=str, default="rank_sum", choices=["rank_sum", "size_aware"])
    parser.add_argument("--lambda_pop", type=int, default=None)
    parser.add_argument("--population_strategy", type=str, default="all", choices=["all", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_list", nargs="+", default=None, help="多任务实验协议，例如: sst2 qnli mnli")
    parser.add_argument("--model_list", nargs="+", default=None, help="多骨干实验协议，例如: roberta-base")
    parser.add_argument("--export_csv", type=str, default="benchmark_results.csv")
    parser.add_argument("--ddp", action="store_true", help="是否启用 torchrun/DistributedDataParallel")
    parser.add_argument("--ddp_backend", type=str, default="nccl", help="DDP backend，通常是 nccl")
    return parser.parse_args()


def run_protocol_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    论文协议批量运行骨架：
    - 支持多任务 × 多骨干 × 多方法
    - 导出 CSV 方便直接填表（主结果/效率/消融）
    """
    tasks = args.task_list if args.task_list else [args.task_name]
    models = args.model_list if args.model_list else [args.model_name]
    all_results: List[Dict[str, Any]] = []

    for task_name in tasks:
        for model_name in models:
            train_loader, val_loader, base_model, tokenizer = setup_data_and_model(
                task_name=task_name,
                model_name=model_name,
                batch_size=args.batch_size,
                max_length=args.max_length,
                task_type=args.task_type,
                nlg_dataset_name=args.nlg_dataset_name,
                max_target_length=args.max_target_length,
                dataset_cache_dir=args.dataset_cache_dir,
                model_cache_dir=args.model_cache_dir,
                ddp_enabled=args.ddp_enabled,
                rank=args.rank,
                world_size=args.world_size,
                seed=args.seed,
            )
            planned_total_steps = (
                args.max_train_steps if args.max_train_steps is not None else args.epochs * len(train_loader)
            )
            for method in args.methods:
                method_model = copy.deepcopy(base_model)
                method_model, controller, meta = peft_factory(
                    model=method_model,
                    method_name=method,
                    target_rank=args.target_rank,
                    total_steps=planned_total_steps,
                    adalora_delta_t=args.adalora_delta_t,
                )
                res = run_training_loop(
                    model=method_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    method_name=method,
                    controller=controller,
                    task_type=args.task_type,
                    tokenizer=tokenizer,
                    generation_max_new_tokens=args.generation_max_new_tokens,
                    nlg_eval_max_samples=args.nlg_eval_max_samples,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    warmup_ratio=args.warmup_ratio,
                    T_es=args.T_es,
                    mini_val_k=args.mini_val_k,
                    max_train_steps=args.max_train_steps,
                    log_dir=args.log_dir,
                    use_wandb=args.use_wandb,
                    wandb_project=args.wandb_project,
                    is_main_process=args.is_main_process,
                    ddp_enabled=args.ddp_enabled,
                    local_rank=args.local_rank,
                    lambda_c=args.lambda_c,
                    complexity_mode=args.complexity_mode,
                    lambda_pop=args.lambda_pop,
                    population_strategy=args.population_strategy,
                    random_seed=args.seed,
                )
                res.update(
                    {
                        "task": task_name,
                        "backbone": model_name,
                        "trainable_params": meta["trainable_params"],
                    }
                )
                if args.is_main_process:
                    all_results.append(res)

    if args.is_main_process:
        with open(args.export_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "task",
                    "backbone",
                    "method",
                    "trainable_params",
                    "best_val_accuracy",
                    "peak_memory_mb",
                    "avg_active_rank",
                    "total_train_time_sec",
                ],
            )
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)

    return all_results


if __name__ == "__main__":
    args = parse_args()
    # 统一 HuggingFace 缓存到当前工程目录，避免默认写入用户主目录。
    os.makedirs(args.dataset_cache_dir, exist_ok=True)
    os.makedirs(args.model_cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", os.path.abspath(args.model_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.abspath(args.model_cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath(args.dataset_cache_dir))

    # 从 torchrun 环境变量读取分布式信息
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    args.ddp_enabled = bool(args.ddp) or env_world_size > 1
    args.rank = env_rank
    args.world_size = env_world_size if args.ddp_enabled else 1
    args.local_rank = env_local_rank if args.ddp_enabled else 0
    args.is_main_process = args.rank == 0

    try:
        if args.ddp_enabled and not dist.is_initialized():
            dist.init_process_group(backend=args.ddp_backend)

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        results = run_protocol_grid(args)

        if args.is_main_process:
            print("\n=== Benchmark Summary ===")
            print(f"{'task':<8} {'backbone':<16} {'method':<12} {'best_acc':<10} {'peak_mem_mb':<12} {'avg_rank':<12} {'time_sec':<12}")
            for r in results:
                print(
                    f"{r.get('task', args.task_name):<8} "
                    f"{r.get('backbone', args.model_name):<16} "
                    f"{r['method']:<12} "
                    f"{r['best_val_accuracy']:<10.4f} "
                    f"{r['peak_memory_mb']:<12.2f} "
                    f"{str(r['avg_active_rank']):<12} "
                    f"{r['total_train_time_sec']:<12.2f}"
                )
    finally:
        if args.ddp_enabled and dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
