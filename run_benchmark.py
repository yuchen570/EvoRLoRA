import argparse
import copy
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model

from rank_evolution_controller import RankEvolutionController
from train_integration import inject_evo_lora, train_evo_lora_step


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


def extract_features_and_labels(batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    features = {k: v for k, v in batch.items() if k != "labels"}
    labels = batch["labels"]
    return features, labels


def setup_data_and_model(
    task_name: str = "sst2",
    model_name: str = "roberta-base",
    batch_size: int = 16,
    max_length: int = 128,
) -> Tuple[DataLoader, DataLoader, nn.Module]:
    dataset = load_dataset("glue", task_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sentence_keys = {
        "sst2": ("sentence", None),
    }
    if task_name not in sentence_keys:
        raise ValueError(f"当前脚本只内置了 {list(sentence_keys.keys())} 的字段映射，请扩展后再用: {task_name}")

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
    train_loader = DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(tokenized["validation"], batch_size=batch_size, shuffle=False, collate_fn=collator)

    num_labels = 2 if task_name == "sst2" else None
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return train_loader, val_loader, base_model


def peft_factory(
    model: nn.Module,
    method_name: str,
    target_rank: int = 8,
) -> Tuple[nn.Module, Optional[RankEvolutionController], Dict[str, Any]]:
    target_modules = ["query", "value"]
    controller: Optional[RankEvolutionController] = None

    if method_name == "lora":
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=target_rank,
            lora_alpha=2 * target_rank,
            lora_dropout=0.1,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, config)

    elif method_name == "adalora":
        config = AdaLoraConfig(
            task_type=TaskType.SEQ_CLS,
            init_r=target_rank * 2,
            target_r=target_rank,
            lora_alpha=2 * target_rank,
            lora_dropout=0.1,
            target_modules=target_modules,
        )
        model = get_peft_model(model, config)

    elif method_name == "evorank":
        controller = inject_evo_lora(
            model=model,
            target_modules=target_modules,
            layer_kwargs={"r_max": 16, "r_init": target_rank, "lora_alpha": 2.0 * target_rank},
            controller_kwargs={"rho": 0.9, "p_g": 0.8, "p_p": 0.1, "H_g": 2, "H_p": 3, "cooldown_steps": 2},
        )

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
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    ce_loss = nn.CrossEntropyLoss()

    total_train_steps = epochs * len(train_loader)
    warmup_steps = int(total_train_steps * warmup_ratio)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, method_name))

    wandb_run = None
    if use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(project=wandb_project, name=method_name, config={"method": method_name})
        except Exception:
            wandb_run = None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    mini_val_batches: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]] = []
    for i, vb in enumerate(val_loader):
        if i >= mini_val_k:
            break
        vb = batch_to_device(vb, device)
        feats, labels = extract_features_and_labels(vb)
        mini_val_batches.append((feats, labels))

    start_time = time.perf_counter()
    global_step = 0
    best_val_acc = 0.0
    train_loss_ema = None
    ema_beta = 0.95

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            features, labels = extract_features_and_labels(batch)

            if method_name == "evorank":
                if controller is None:
                    raise ValueError("evorank 方法必须传入 controller")
                out = train_evo_lora_step(
                    model=model,
                    controller=controller,
                    optimizer=optimizer,
                    train_batch=(features, labels),
                    val_batch=mini_val_batches,
                    loss_fn=ce_loss,
                    step=global_step,
                    warmup_steps=warmup_steps,
                    T_es=T_es,
                )
                train_loss = float(out["train_loss"])
                avg_active_rank = float(
                    sum(layer.get_active_rank() for layer in controller.layers.values()) / max(len(controller.layers), 1)
                )
            else:
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = ce_loss(logits, labels)
                loss.backward()
                optimizer.step()
                if method_name == "adalora":
                    _adalora_post_step_update(model, global_step)
                train_loss = float(loss.detach().item())
                avg_active_rank = float("nan")

            train_loss_ema = train_loss if train_loss_ema is None else (ema_beta * train_loss_ema + (1 - ema_beta) * train_loss)
            writer.add_scalar("train/loss", train_loss, global_step)
            writer.add_scalar("train/loss_ema", train_loss_ema, global_step)
            if method_name == "evorank":
                writer.add_scalar("train/active_rank_mean", avg_active_rank, global_step)
            if wandb_run is not None:
                wandb_run.log({"train/loss": train_loss, "train/loss_ema": train_loss_ema, "step": global_step})

            global_step += 1
            if max_train_steps is not None and global_step >= max_train_steps:
                break

        # 每个 epoch 在完整验证集上评估 Accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for vb in val_loader:
                vb = batch_to_device(vb, device)
                features, labels = extract_features_and_labels(vb)
                logits = model(features)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        val_acc = correct / max(total, 1)
        best_val_acc = max(best_val_acc, val_acc)

        print(
            f"[{method_name}] epoch={epoch + 1}/{epochs} "
            f"step={global_step} val_acc={val_acc:.4f} best={best_val_acc:.4f}"
        )
        writer.add_scalar("val/accuracy", val_acc, epoch)
        if wandb_run is not None:
            wandb_run.log({"val/accuracy": val_acc, "epoch": epoch + 1, "step": global_step})

        if max_train_steps is not None and global_step >= max_train_steps:
            break

    total_time = time.perf_counter() - start_time
    peak_mem_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if torch.cuda.is_available()
        else 0.0
    )

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
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default="runs/benchmark")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="evorank-benchmark")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_loader, val_loader, base_model = setup_data_and_model(
        task_name=args.task_name,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    results: List[Dict[str, Any]] = []
    for method in args.methods:
        # 每个方法从同一个初始权重出发，保证公平对比。
        method_model = copy.deepcopy(base_model)
        method_model, controller, meta = peft_factory(
            model=method_model,
            method_name=method,
            target_rank=args.target_rank,
        )
        print(
            f"[{method}] trainable_params={meta['trainable_params']:,} "
            f"(预算参考: LoRA r={args.target_rank})"
        )
        res = run_training_loop(
            model=method_model,
            train_loader=train_loader,
            val_loader=val_loader,
            method_name=method,
            controller=controller,
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
        )
        results.append(res)

    print("\n=== Benchmark Summary ===")
    print(f"{'method':<12} {'best_acc':<10} {'peak_mem_mb':<12} {'avg_rank':<12} {'time_sec':<12}")
    for r in results:
        print(
            f"{r['method']:<12} "
            f"{r['best_val_accuracy']:<10.4f} "
            f"{r['peak_memory_mb']:<12.2f} "
            f"{str(r['avg_active_rank']):<12} "
            f"{r['total_train_time_sec']:<12.2f}"
        )
