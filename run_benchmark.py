import argparse
import copy
import csv
import json
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

from adalora_utils import adalora_update_and_allocate, compute_adalora_orthogonal_loss, get_adalora_orth_reg_weight, unwrap_inner_from_training_model
from lora_ga_init import apply_lora_ga_init_to_peft, run_lora_ga_init_pipeline
from rank_evolution_controller import RankEvolutionController
from sora_inject import SparseAdamW, inject_sora
from train_integration import inject_evo_lora, train_evo_lora_step

from glue_metrics import collect_nlu_predictions, compute_glue_primary_metric, glue_primary_metric_key

from torch.nn.parallel import DistributedDataParallel as DDP

# HuggingFace `datasets` 中 `load_dataset("glue", <config>)` 的全集（sentence 字段与验证 split 约定）。
# 参考：https://huggingface.co/datasets/glue
GLUE_TASK_SENTENCE_KEYS: Dict[str, Tuple[str, Optional[str]]] = {
    "ax": ("premise", "hypothesis"),
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
# STS-B 为回归标签；验证主指标为 (Pearson+Spearman)/2，见 glue_metrics.py。
GLUE_REGRESSION_TASKS = frozenset({"stsb"})


def glue_validation_split(task_name: str) -> str:
    if task_name == "mnli":
        return "validation_matched"
    return "validation"


def nlu_is_glue_regression(task_name: Optional[str]) -> bool:
    return task_name is not None and task_name in GLUE_REGRESSION_TASKS


class DictFeatureClassifier(nn.Module):
    """
    适配 train_evo_lora_step 的输入约定：
    - 输入是特征字典（input_ids/attention_mask/...）
    - 输出为分类 logits 或回归标量（STS-B 等为 shape (B,1) 的 logits）
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
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], nn.Module, AutoTokenizer]:
    os.makedirs(dataset_cache_dir, exist_ok=True)
    os.makedirs(model_cache_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir)

    if task_type == "nlu":
        dataset = load_dataset("glue", task_name, cache_dir=dataset_cache_dir)

        if task_name not in GLUE_TASK_SENTENCE_KEYS:
            raise ValueError(
                f"未知 GLUE 子集 task_name={task_name!r}。已支持: {sorted(GLUE_TASK_SENTENCE_KEYS.keys())}"
            )

        sentence1_key, sentence2_key = GLUE_TASK_SENTENCE_KEYS[task_name]

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
        val_split_name = glue_validation_split(task_name)

        val_loader_eval_full: Optional[DataLoader] = None
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
            # GLUE 官方主指标（MCC/F1 等）需全验证集；与 NLG 一致，仅 rank0 用该 loader 做评估。
            val_loader_eval_full = DataLoader(
                tokenized[val_split_name], batch_size=batch_size, shuffle=False, collate_fn=collator
            )
        else:
            train_loader = DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True, collate_fn=collator)
            val_loader = DataLoader(tokenized[val_split_name], batch_size=batch_size, shuffle=False, collate_fn=collator)

        if task_name == "stsb":
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1,
                problem_type="regression",
                cache_dir=model_cache_dir,
            )
        else:
            label_feature = dataset["train"].features.get("label", None)
            if label_feature is not None and hasattr(label_feature, "num_classes") and label_feature.num_classes is not None:
                num_labels = int(label_feature.num_classes)
            else:
                num_labels = len(set(dataset["train"]["label"]))
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                cache_dir=model_cache_dir,
            )
        return train_loader, val_loader, val_loader_eval_full, base_model, tokenizer

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
            # DDP 下验证集被切分；ROUGE 需在 rank0 上对完整验证集评估（他卡 barrier 等待）。
            val_loader_eval_full = DataLoader(
                tokenized[val_split_name], batch_size=batch_size, shuffle=False, collate_fn=collator
            )
        else:
            train_loader = DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True, collate_fn=collator)
            val_loader = DataLoader(tokenized[val_split_name], batch_size=batch_size, shuffle=False, collate_fn=collator)
            val_loader_eval_full = None

        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=model_cache_dir)
        return train_loader, val_loader, val_loader_eval_full, base_model, tokenizer

    raise ValueError(f"未知 task_type: {task_type}")


def peft_factory(
    model: nn.Module,
    method_name: str,
    target_rank: int = 8,
    total_steps: Optional[int] = None,
    adalora_delta_t: int = 200,
    train_loader: Optional[DataLoader] = None,
    lora_ga_batches: int = 8,
    task_type: str = "nlu",
    lora_ga_device: Optional[torch.device] = None,
    is_main_process: bool = True,
    ddp_enabled: bool = False,
    adalora_init_r: Optional[int] = None,
    adalora_tinit: Optional[int] = None,
    adalora_tfinal: Optional[int] = None,
    adalora_orth_reg_weight: float = 0.1,
    nlu_regression: bool = False,
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
        # AdaLoRA：预算调度 + RankAllocator（步后 update_and_allocate）+ 正交正则（本脚本在 loss 上显式加入，见 run_training_loop）。
        planned_steps = max(int(total_steps or 1000), 1)
        if adalora_tinit is None:
            tinit = max(int(0.1 * planned_steps), 1)
        else:
            tinit = max(int(adalora_tinit), 1)
        if adalora_tfinal is None:
            tfinal = max(int(0.8 * planned_steps), tinit + 1)
        else:
            tfinal = max(int(adalora_tfinal), tinit + 1)
        if tinit >= planned_steps - tfinal:
            raise ValueError(
                f"AdaLoRA 调度无效：需满足 tinit < total_step - tfinal，当前 total_step={planned_steps}, "
                f"tinit={tinit}, tfinal={tfinal}。请减小 --adalora_tinit/--adalora_tfinal 或增大训练步数。"
            )
        init_r_val = int(adalora_init_r) if adalora_init_r is not None else target_rank * 2
        if init_r_val < target_rank:
            raise ValueError("adalora_init_r 应 >= target_rank（target_r）")

        adalora_kw: Dict[str, Any] = dict(
            task_type=peft_task_type,
            init_r=init_r_val,
            target_r=target_rank,
            lora_alpha=2 * target_rank,
            lora_dropout=0.1,
            target_modules=target_modules,
            bias="none",
            beta1=0.85,
            beta2=0.85,
            total_step=planned_steps,
            tinit=tinit,
            tfinal=tfinal,
            deltaT=adalora_delta_t,
        )
        try:
            from dataclasses import fields

            _adalora_field_names = {f.name for f in fields(AdaLoraConfig)}
        except Exception:
            _adalora_field_names = set()
        if "orth_reg_weight" in _adalora_field_names:
            adalora_kw["orth_reg_weight"] = float(adalora_orth_reg_weight)
        config = AdaLoraConfig(**adalora_kw)
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

    elif method_name == "lora-ga":
        if train_loader is None:
            raise ValueError("method_name='lora-ga' 时必须传入 train_loader")
        ga_dev = lora_ga_device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        ga_loss_fn: Optional[nn.Module] = None
        if task_type == "nlu" and nlu_regression:
            ga_loss_fn = nn.MSELoss()
        init_by_key = run_lora_ga_init_pipeline(
            base_model=model,
            train_loader=train_loader,
            target_modules=target_modules,
            lora_r=target_rank,
            lora_ga_batches=lora_ga_batches,
            task_type=task_type,
            device=ga_dev,
            is_main_process=is_main_process,
            ddp_enabled=ddp_enabled,
            loss_fn=ga_loss_fn,
        )
        config = LoraConfig(
            task_type=peft_task_type,
            r=target_rank,
            lora_alpha=2 * target_rank,
            lora_dropout=0.1,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, config)
        tgt = next(model.parameters()).device
        apply_lora_ga_init_to_peft(model, init_by_key, target_device=tgt)

    elif method_name == "sora":
        inject_sora(
            model=model,
            target_modules=target_modules,
            r=target_rank,
            lora_alpha=float(2 * target_rank),
            lora_dropout=0.1,
        )

    elif method_name == "mtl-lora":
        raise NotImplementedError(
            "mtl-lora 需联合多任务数据与 task id；当前 --task_list 为逐任务串行。见 README / MTL-LoRA 官方仓库。"
        )
    else:
        raise ValueError(f"未知 method_name: {method_name}")

    wrapped_model = DictFeatureClassifier(model)
    trainable_params = count_trainable_params(wrapped_model)
    meta = {"trainable_params": trainable_params}
    return wrapped_model, controller, meta


def _adalora_post_step_update(model: nn.Module, global_step: int) -> None:
    """PEFT AdaLora 步后 RankAllocator / mask 更新（兼容 DDP + DictFeatureClassifier）。"""
    adalora_update_and_allocate(model, global_step)


def _unwrap_training_module(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _unwrap_for_save(model: nn.Module) -> nn.Module:
    """剥离 DDP 后取 DictFeatureClassifier.inner，避免 state_dict key 带 module. 前缀。"""
    m = _unwrap_training_module(model)
    return getattr(m, "inner", m)


def _state_dict_cpu(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def _save_checkpoint_pt(
    path: str,
    inner: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    global_step: int,
    epoch: int,
    best_val: float,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "global_step": global_step,
        "epoch": epoch,
        "best_val_accuracy": best_val,
        "model": _state_dict_cpu(inner),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }
    torch.save(payload, path)


def _save_final_artifact(
    inner: nn.Module,
    final_dir: str,
    method_name: str,
    task_type: str,
    task_name: Optional[str],
    tokenizer: Optional[Any],
    best_val_accuracy: float,
    global_step: int,
) -> None:
    os.makedirs(final_dir, exist_ok=True)
    meta: Dict[str, Any] = {
        "method": method_name,
        "task_type": task_type,
        "best_val_accuracy": float(best_val_accuracy),
        "global_step": int(global_step),
    }
    if task_name is not None:
        meta["task_name"] = task_name
    with open(os.path.join(final_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if hasattr(inner, "save_pretrained"):
        inner.save_pretrained(final_dir)
    else:
        torch.save(_state_dict_cpu(inner), os.path.join(final_dir, "model_state.pt"))
    if tokenizer is not None:
        tokenizer.save_pretrained(final_dir)


def _run_verify_samples(
    model: nn.Module,
    val_loader: DataLoader,
    task_type: str,
    device: torch.device,
    tokenizer: Optional[Any],
    verify_n_samples: int,
    generation_max_new_tokens: int,
    glue_task_name: Optional[str] = None,
) -> None:
    if verify_n_samples <= 0:
        return
    infer_model = _unwrap_training_module(model)
    infer_model.eval()
    with torch.no_grad():
        if task_type == "nlu":
            for vb in val_loader:
                vb = batch_to_device(vb, device)
                features, labels = extract_features_and_labels(vb, task_type=task_type)
                logits = infer_model(features)
                n = min(verify_n_samples, int(labels.size(0)))
                if nlu_is_glue_regression(glue_task_name):
                    pred_scores = logits.squeeze(-1)
                    for i in range(n):
                        g = float(labels[i].item())
                        p = float(pred_scores[i].item())
                        print(f"[verify] sample {i} [Gold]={g:.4f} [Pred]={p:.4f}")
                else:
                    preds = torch.argmax(logits, dim=-1)
                    for i in range(n):
                        g = int(labels[i].item())
                        p = int(preds[i].item())
                        print(f"[verify] sample {i} [Gold]={g} [Pred]={p}")
                break
        elif task_type == "nlg":
            if tokenizer is None:
                return
            gen_model = _unwrap_for_save(model)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            for vb in val_loader:
                vb = batch_to_device(vb, device)
                input_ids = vb["input_ids"]
                attention_mask = vb.get("attention_mask", None)
                gen_kwargs: Dict[str, Any] = dict(input_ids=input_ids, max_new_tokens=generation_max_new_tokens)
                if attention_mask is not None:
                    gen_kwargs["attention_mask"] = attention_mask
                gen_ids = gen_model.generate(**gen_kwargs)
                pred = tokenizer.decode(gen_ids[0].cpu(), skip_special_tokens=True)
                labels = vb["labels"][0].clone()
                labels[labels == -100] = pad_id
                gold = tokenizer.decode(labels.cpu(), skip_special_tokens=True)
                print(f"[verify] [Gold] (trunc): {gold[:240]}")
                print(f"[verify] [Pred] (trunc): {pred[:240]}")
                break
        else:
            raise ValueError(f"未知 task_type: {task_type}")


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
    val_loader_eval_full: Optional[DataLoader] = None,
    sora_sparse_lambda: float = 1e-3,
    sora_sparse_lambda_2: float = 1e-3,
    sora_lambda_warmup_steps: int = 0,
    task_name: Optional[str] = None,
    checkpoint_root: Optional[str] = None,
    save_steps: int = 0,
    save_every_epoch: bool = False,
    save_final_model: bool = True,
    verify_n_samples: int = 2,
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
    if method_name == "sora":
        # 官方 SoRA：gate 参数使用独立的 SparseAdamW（近端梯度），其余参数使用标准 AdamW。
        _non_gate = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith(".gate")]
        _gate = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith(".gate")]
        optimizer = AdamW(_non_gate, lr=lr, weight_decay=weight_decay)
        sparse_optimizer = SparseAdamW(_gate, lr=lr, sparse_lambda=sora_sparse_lambda_2, weight_decay=0.0)
    else:
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        sparse_optimizer = None
    if task_type == "nlu":
        if nlu_is_glue_regression(task_name):

            def nlu_reg_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                return F.mse_loss(logits.squeeze(-1), labels.float())

            loss_fn = nlu_reg_loss
        else:
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
    if sparse_optimizer is not None:
        sparse_lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=sparse_optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
        )
    else:
        sparse_lr_scheduler = None
    writer = SummaryWriter(log_dir=os.path.join(log_dir, method_name)) if is_main_process else None

    if checkpoint_root and is_main_process:
        os.makedirs(checkpoint_root, exist_ok=True)

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
                if sparse_optimizer is not None:
                    sparse_optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = loss_fn(logits, labels)
                if method_name == "adalora":
                    inner = unwrap_inner_from_training_model(model)
                    ow = get_adalora_orth_reg_weight(inner)
                    if ow > 0:
                        loss = loss + ow * compute_adalora_orthogonal_loss(inner)
                if method_name == "sora":
                    lam = float(sora_sparse_lambda)
                    if sora_lambda_warmup_steps > 0:
                        lam *= min(1.0, float(global_step + 1) / float(sora_lambda_warmup_steps))
                    # 官方 SoRA：L1 Loss 除以 gate 总元素数做归一化
                    l1_penalty = sum(
                        p.abs().sum() for n, p in model.named_parameters() if n.endswith(".gate")
                    )
                    gate_numel = sum(
                        p.numel() for n, p in model.named_parameters() if n.endswith(".gate")
                    )
                    loss = loss + lam * l1_penalty / max(gate_numel, 1)
                loss.backward()
                optimizer.step()
                if sparse_optimizer is not None:
                    sparse_optimizer.step()
                if method_name == "adalora":
                    _adalora_post_step_update(model, global_step)
                train_loss = float(loss.detach().item())
                avg_active_rank = float("nan")

            # 无论哪条路径，optimizer.step() 都已在本步完成，此处统一推进学习率调度。
            lr_scheduler.step()
            if sparse_lr_scheduler is not None:
                sparse_lr_scheduler.step()
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
            if (
                is_main_process
                and checkpoint_root
                and save_steps > 0
                and global_step % save_steps == 0
            ):
                inner_ckpt = _unwrap_for_save(model)
                _save_checkpoint_pt(
                    os.path.join(checkpoint_root, f"checkpoint_step_{global_step}.pt"),
                    inner_ckpt,
                    optimizer,
                    lr_scheduler,
                    global_step,
                    epoch,
                    best_val_acc,
                )
            if max_train_steps is not None and global_step >= max_train_steps:
                break

        # 每个 epoch 在完整验证集上评估指标
        model.eval()

        if task_type == "nlu":
            if task_name is None:
                raise ValueError("NLU（GLUE）评估需要传入 task_name")
            if ddp_enabled and dist.is_available() and dist.is_initialized():
                dist.barrier()
            val_metric = 0.0
            mkey = glue_primary_metric_key(task_name)
            regression = nlu_is_glue_regression(task_name)
            if is_main_process:
                ev_loader = val_loader_eval_full if val_loader_eval_full is not None else val_loader
                eval_model = _unwrap_training_module(model)
                y_pred, y_true = collect_nlu_predictions(eval_model, ev_loader, device, regression=regression)
                val_metric = compute_glue_primary_metric(task_name, y_pred, y_true)
            if ddp_enabled and dist.is_available() and dist.is_initialized():
                m_tensor = torch.tensor([val_metric], device=device, dtype=torch.float64)
                dist.broadcast(m_tensor, src=0)
                val_metric = float(m_tensor.item())
                dist.barrier()
            best_val_acc = max(best_val_acc, val_metric)
            if is_main_process:
                print(
                    f"[{method_name}] epoch={epoch + 1}/{epochs} "
                    f"step={global_step} val_{mkey}={val_metric:.4f} best={best_val_acc:.4f}"
                )
                if writer is not None:
                    writer.add_scalar(f"val/{mkey}", val_metric, epoch)
                if wandb_run is not None:
                    wandb_run.log(
                        {f"val/{mkey}": val_metric, "epoch": epoch + 1, "step": global_step}
                    )

        else:
            if tokenizer is None:
                raise ValueError("task_type='nlg' 时必须传入 tokenizer")
            # DDP 下用 barrier 对齐：rank0 在完整验证集上算 ROUGE，避免 DistributedSampler 子集有偏。
            if ddp_enabled and dist.is_available() and dist.is_initialized():
                dist.barrier()

            val_metric = 0.0
            if is_main_process:
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

                eval_loader = val_loader_eval_full if val_loader_eval_full is not None else val_loader

                preds_text: List[str] = []
                refs_text: List[str] = []
                sample_count = 0

                with torch.no_grad():
                    for vb in eval_loader:
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
                print(
                    f"[{method_name}] epoch={epoch + 1}/{epochs} "
                    f"step={global_step} val_rougeL={val_metric:.4f} best={best_val_acc:.4f}"
                )
                if writer is not None:
                    writer.add_scalar("val/rougeL", val_metric, epoch)
                if wandb_run is not None:
                    wandb_run.log({"val/rougeL": val_metric, "epoch": epoch + 1, "step": global_step})

            if ddp_enabled and dist.is_available() and dist.is_initialized():
                dist.barrier()

        if is_main_process and checkpoint_root:
            rec: Dict[str, Any] = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "val_metric": float(val_metric),
                "best_val": float(best_val_acc),
                "train_loss_ema": float(train_loss_ema) if train_loss_ema is not None else None,
            }
            if task_type == "nlu" and task_name is not None:
                rec["glue_metric"] = glue_primary_metric_key(task_name)
            with open(os.path.join(checkpoint_root, "metrics.jsonl"), "a", encoding="utf-8") as mf:
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if save_every_epoch:
                inner_ep = _unwrap_for_save(model)
                _save_checkpoint_pt(
                    os.path.join(checkpoint_root, f"checkpoint_epoch_{epoch + 1}.pt"),
                    inner_ep,
                    optimizer,
                    lr_scheduler,
                    global_step,
                    epoch,
                    best_val_acc,
                )

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

    model.eval()
    if is_main_process and checkpoint_root and save_final_model:
        final_dir = os.path.join(checkpoint_root, "final")
        _save_final_artifact(
            _unwrap_for_save(model),
            final_dir,
            method_name,
            task_type,
            task_name,
            tokenizer,
            best_val_acc,
            global_step,
        )
    if is_main_process and verify_n_samples > 0:
        _run_verify_samples(
            model,
            val_loader,
            task_type,
            device,
            tokenizer,
            verify_n_samples,
            generation_max_new_tokens,
            glue_task_name=task_name,
        )

    if ddp_enabled and dist.is_available() and dist.is_initialized():
        dist.barrier()

    artifact_dir_str = checkpoint_root or ""
    final_dir_str = (
        os.path.join(checkpoint_root, "final")
        if (checkpoint_root and save_final_model)
        else ""
    )
    val_metric_key_str = ""
    if task_type == "nlu" and task_name is not None:
        try:
            val_metric_key_str = glue_primary_metric_key(task_name)
        except ValueError:
            val_metric_key_str = ""
    elif task_type == "nlg":
        val_metric_key_str = "rougeL"

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
        "artifact_dir": artifact_dir_str,
        "final_dir": final_dir_str,
        "val_metric_key": val_metric_key_str,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EvoRank-LoRA GLUE benchmark runner")
    parser.add_argument(
        "--task_name",
        type=str,
        default="sst2",
        help="NLU：`load_dataset('glue', task_name)` 的子集，已支持 ax, cola, sst2, mrpc, qqp, stsb, mnli, qnli, rte, wnli；"
        "验证集使用各任务 GLUE 官方主指标（见 glue_metrics / README）。NLG：占位/命名用。",
    )
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
    parser.add_argument("--adalora_init_r", type=int, default=None, help="AdaLoRA init_r，默认 2*target_rank")
    parser.add_argument("--adalora_tinit", type=int, default=None, help="AdaLoRA 初始满秩阶段步数 tinit，默认 floor(0.1*total_step)")
    parser.add_argument("--adalora_tfinal", type=int, default=None, help="AdaLoRA 末尾不调秩阶段步数 tfinal，默认 floor(0.8*total_step)")
    parser.add_argument(
        "--adalora_orth_reg_weight",
        type=float,
        default=0.1,
        help="AdaLoRA 正交正则（与 PEFT/论文一致；本脚本因只取 logits 需显式加项）",
    )
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
    parser.add_argument("--lora_ga_batches", type=int, default=8, help="LoRA-GA 梯度估计用的训练 batch 数（仅前若干个 batch）")
    parser.add_argument("--sora_sparse_lambda", type=float, default=1e-3, help="SoRA：gate L1 惩罚系数")
    parser.add_argument("--sora_lambda_warmup_steps", type=int, default=0, help="SoRA：前若干步将 sparse_lambda 从 0 线性升到设定值；0 表示关闭")
    parser.add_argument("--sora_sparse_lambda_2", type=float, default=1e-3, help="SoRA：gate 近端梯度硬裁剪阈值（Proximal Gradient / Soft-Thresholding），对标官方 SparseAdamW")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts",
        help="每个 task×backbone×method 在此目录下建子目录，写入 metrics.jsonl、可选 checkpoint、final/",
    )
    parser.add_argument("--no_output_dir", action="store_true", help="关闭 output_dir 下所有落盘（metrics/checkpoint/final）")
    parser.add_argument("--save_steps", type=int, default=0, help="每 N 个训练 step 保存 checkpoint_step_*.pt；0 表示不按步保存")
    parser.add_argument("--save_every_epoch", action="store_true", help="每个 epoch 结束保存 checkpoint_epoch_*.pt")
    parser.add_argument("--no_save_final_model", action="store_true", help="训练结束不写入 final/（adapter 或 model_state.pt + tokenizer 等）")
    parser.add_argument(
        "--verify_n_samples",
        type=int,
        default=2,
        help="训练结束后主进程从验证集打印前 K 条 [Gold] vs [Pred]（NLU）或一条生成摘要片段（NLG）；0 关闭",
    )
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

    if args.is_main_process and not args.no_output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for task_name in tasks:
        for model_name in models:
            train_loader, val_loader, val_loader_eval_full, base_model, tokenizer = setup_data_and_model(
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
                if args.ddp_enabled and torch.cuda.is_available():
                    lora_ga_dev = torch.device(f"cuda:{args.local_rank}")
                else:
                    lora_ga_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                method_model, controller, meta = peft_factory(
                    model=method_model,
                    method_name=method,
                    target_rank=args.target_rank,
                    total_steps=planned_total_steps,
                    adalora_delta_t=args.adalora_delta_t,
                    adalora_init_r=args.adalora_init_r,
                    adalora_tinit=args.adalora_tinit,
                    adalora_tfinal=args.adalora_tfinal,
                    adalora_orth_reg_weight=args.adalora_orth_reg_weight,
                    train_loader=train_loader,
                    lora_ga_batches=args.lora_ga_batches,
                    task_type=args.task_type,
                    lora_ga_device=lora_ga_dev,
                    is_main_process=args.is_main_process,
                    ddp_enabled=args.ddp_enabled,
                    nlu_regression=nlu_is_glue_regression(task_name),
                )
                checkpoint_root: Optional[str] = None
                if not args.no_output_dir:
                    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
                    checkpoint_root = os.path.join(args.output_dir, f"{task_name}_{safe_model_name}_{method}")
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
                    val_loader_eval_full=val_loader_eval_full,
                    sora_sparse_lambda=args.sora_sparse_lambda,
                    sora_sparse_lambda_2=args.sora_sparse_lambda_2,
                    sora_lambda_warmup_steps=args.sora_lambda_warmup_steps,
                    task_name=task_name,
                    checkpoint_root=checkpoint_root,
                    save_steps=args.save_steps,
                    save_every_epoch=args.save_every_epoch,
                    save_final_model=not args.no_save_final_model,
                    verify_n_samples=args.verify_n_samples,
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
                    "val_metric_key",
                    "trainable_params",
                    "best_val_accuracy",
                    "peak_memory_mb",
                    "avg_active_rank",
                    "total_train_time_sec",
                    "artifact_dir",
                    "final_dir",
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
            if results:
                mk = (results[0].get("val_metric_key") or "").strip()
                if mk:
                    print(
                        f"[summary] 验证主指标日志键为 val_{mk}；CSV 列名仍为 best_val_accuracy（存同一数值）。"
                    )
            print("\n=== Benchmark Summary ===")
            print(
                f"{'task':<8} {'backbone':<16} {'method':<12} {'best_metric':<12} "
                f"{'peak_mem_mb':<12} {'avg_rank':<10} {'time_sec':<10}"
            )
            for r in results:
                ar = r["avg_active_rank"]
                ar_fmt = f"{float(ar):.4f}" if isinstance(ar, float) else str(ar)
                print(
                    f"{r.get('task', args.task_name):<8} "
                    f"{r.get('backbone', args.model_name):<16} "
                    f"{r['method']:<12} "
                    f"{r['best_val_accuracy']:<12.4f} "
                    f"{r['peak_memory_mb']:<12.2f} "
                    f"{ar_fmt:<10} "
                    f"{r['total_train_time_sec']:<10.2f}"
                )
    finally:
        if args.ddp_enabled and dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
