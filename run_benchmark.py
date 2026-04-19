import argparse
import copy
import csv
import datetime
import json
import os
import random
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
)

from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model

from adalora_utils import (
    adalora_update_and_allocate,
    compute_adalora_orthogonal_loss,
    get_adalora_orth_reg_weight,
    normalize_adalora_schedule,
    unwrap_inner_from_training_model,
)
from rank_evolution_controller import RankEvolutionController
from sora_inject import SparseAdamW, inject_sora
from toplora_inject import inject_toplora
from train_integration import inject_evo_lora, train_evo_lora_step

from glue_metrics import collect_nlu_predictions, compute_glue_primary_metric, glue_primary_metric_key, compute_glue_metrics_dict

from hf_cache_utils import check_local_dataset, resolve_pretrained_model_source

from torch.nn.parallel import DistributedDataParallel as DDP


def _linear_warmup_decay_lr_lambda(num_warmup_steps: int, num_training_steps: int):
    """
    与 transformers.get_linear_schedule_with_warmup 同形的 warmup + 线性衰减，但 warmup 在 step=0 的乘子为 1/W 而非 0。
    HF 的 LambdaLR 在构造时会 step 一次，使首步 lr 乘子为 0：此时 AdamW 仍按梯度更新动量，下一步用极小 lr 更新权重时易数值爆炸（运行证据：DeBERTa+LoRA 在 global_step=1 起 loss/分类头即 NaN）。
    """

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step + 1) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return lr_lambda


def _cosine_warmup_decay_lr_lambda(num_warmup_steps: int, num_training_steps: int):
    """带 warmup 的 cosine 衰减（与 HF 常见公式一致，末尾收敛到 0）。"""

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step + 1) / float(max(1, num_warmup_steps))
        if num_training_steps <= num_warmup_steps:
            return 1.0
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


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


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    payload = {
        "sessionId": "a6bd12",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open("debug-a6bd12.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


def glue_nlu_train_val_splits(dataset, task_name: str) -> Tuple[str, str]:
    """
    返回 (train_split, val_split)。
    - mnli：train + validation_matched
    - 有 train：train + validation（或 test）
    - 无 train 仅有 test/validation：两者复用同一 split（若金标全为 -1 如 HF glue/ax，则由 _assert_glue_split_has_gold_labels 拒绝）
    """
    keys = set(dataset.keys())
    if task_name == "mnli":
        val = "validation_matched" if "validation_matched" in keys else "validation"
        if "train" not in keys:
            raise ValueError(f"GLUE mnli 缺少 train split，现有: {sorted(keys)}")
        if val not in keys:
            raise ValueError(f"GLUE mnli 缺少 {val} split，现有: {sorted(keys)}")
        return "train", val
    if "train" in keys:
        if "validation" in keys:
            return "train", "validation"
        if "test" in keys:
            return "train", "test"
        raise ValueError(f"GLUE {task_name} 有 train 但无 validation/test：{sorted(keys)}")
    if "validation" in keys:
        return "validation", "validation"
    if "test" in keys:
        return "test", "test"
    raise ValueError(f"GLUE {task_name} 无可用 split（需含 train、validation 或 test 之一）：{sorted(keys)}")


def _assert_glue_split_has_gold_labels(dataset, split_name: str, task_name: str) -> None:
    """HuggingFace `glue`/`ax` 的 test 集标签全为 -1（金标不公开），不能用于 CE 训练或主指标。"""
    labs = dataset[split_name]["label"]
    if not labs:
        return
    if all(int(y) == -1 for y in labs):
        raise ValueError(
            f"GLUE 任务 {task_name!r} 的 split {split_name!r} 在 HuggingFace `datasets` 中标签全部为 -1（不公开金标），"
            "无法进行有监督训练或验证（CrossEntropyLoss 需要有效类别下标）。"
            "请从 `--task_name` / `--task_list` 中移除 ax，或先在 `mnli` 上训练后再按官方流程在本地评估 AX。"
        )


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


def _ddp_is_active() -> bool:
    return bool(dist.is_available() and dist.is_initialized())


def _all_gather_object(local_obj: Any) -> List[Any]:
    if not _ddp_is_active():
        return [local_obj]
    world = dist.get_world_size()
    gathered: List[Any] = [None for _ in range(world)]
    dist.all_gather_object(gathered, local_obj)
    return gathered


class _DistributedEvalSampler(Sampler[int]):
    """无重复、无补齐的顺序分片 sampler，用于 DDP 评估阶段。"""

    def __init__(self, dataset, num_replicas: int, rank: int) -> None:
        self.dataset = dataset
        self.num_replicas = max(int(num_replicas), 1)
        self.rank = int(rank)
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(f"invalid rank={self.rank}, num_replicas={self.num_replicas}")

    def __iter__(self):
        total = len(self.dataset)
        return iter(range(self.rank, total, self.num_replicas))

    def __len__(self) -> int:
        total = len(self.dataset)
        if total <= self.rank:
            return 0
        return (total - self.rank + self.num_replicas - 1) // self.num_replicas


def _collect_nlu_predictions_distributed(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    regression: bool,
    ddp_enabled: bool,
    is_main_process: bool,
) -> Tuple[List[float], List[float]]:
    y_pred, y_true = collect_nlu_predictions(model, data_loader, device, regression=regression)
    pred_list = y_pred.reshape(-1).tolist()
    true_list = y_true.reshape(-1).tolist()
    if not ddp_enabled or not _ddp_is_active():
        return pred_list, true_list
    payload = {"pred": pred_list, "true": true_list}
    gathered = _all_gather_object(payload)
    if not is_main_process:
        return [], []
    merged_pred: List[float] = []
    merged_true: List[float] = []
    for item in gathered:
        if not isinstance(item, dict):
            continue
        merged_pred.extend(item.get("pred", []))
        merged_true.extend(item.get("true", []))
    return merged_pred, merged_true


@torch.no_grad()
def _collect_nlg_text_pairs_local(
    gen_model: nn.Module,
    data_loader: DataLoader,
    tokenizer: AutoTokenizer,
    device: torch.device,
    generation_max_new_tokens: int,
    sample_cap: int,
) -> Tuple[List[str], List[str]]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    preds_text: List[str] = []
    refs_text: List[str] = []
    for vb in data_loader:
        vb = batch_to_device(vb, device)
        input_ids = vb["input_ids"]
        attention_mask = vb.get("attention_mask", None)
        labels = vb["labels"]

        gen_kwargs: Dict[str, Any] = dict(input_ids=input_ids, max_new_tokens=generation_max_new_tokens)
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        gen_ids = gen_model.generate(**gen_kwargs)
        pred_batch = tokenizer.batch_decode(gen_ids.detach().cpu(), skip_special_tokens=True)

        labels_ids = labels.clone()
        labels_ids[labels_ids == -100] = pad_id
        ref_batch = tokenizer.batch_decode(labels_ids.detach().cpu(), skip_special_tokens=True)

        preds_text.extend(pred_batch)
        refs_text.extend(ref_batch)
        if len(preds_text) >= sample_cap:
            break
    if len(preds_text) > sample_cap:
        preds_text = preds_text[:sample_cap]
        refs_text = refs_text[:sample_cap]
    return preds_text, refs_text


def _collect_nlg_text_pairs_distributed(
    gen_model: nn.Module,
    data_loader: DataLoader,
    tokenizer: AutoTokenizer,
    device: torch.device,
    generation_max_new_tokens: int,
    nlg_eval_max_samples: int,
    ddp_enabled: bool,
    is_main_process: bool,
) -> Tuple[List[str], List[str]]:
    if ddp_enabled and _ddp_is_active():
        local_cap = max(1, int(math.ceil(float(nlg_eval_max_samples) / float(dist.get_world_size()))))
    else:
        local_cap = nlg_eval_max_samples
    local_pred, local_ref = _collect_nlg_text_pairs_local(
        gen_model=gen_model,
        data_loader=data_loader,
        tokenizer=tokenizer,
        device=device,
        generation_max_new_tokens=generation_max_new_tokens,
        sample_cap=local_cap,
    )
    if not ddp_enabled or not _ddp_is_active():
        return local_pred[:nlg_eval_max_samples], local_ref[:nlg_eval_max_samples]

    gathered = _all_gather_object({"pred": local_pred, "ref": local_ref})
    if not is_main_process:
        return [], []
    merged_pred: List[str] = []
    merged_ref: List[str] = []
    for item in gathered:
        if not isinstance(item, dict):
            continue
        merged_pred.extend(item.get("pred", []))
        merged_ref.extend(item.get("ref", []))
    return merged_pred[:nlg_eval_max_samples], merged_ref[:nlg_eval_max_samples]


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
    dataloader_num_workers: int = 1,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[Union[DataLoader, Dict[str, DataLoader]]], nn.Module, AutoTokenizer]:
    os.makedirs(dataset_cache_dir, exist_ok=True)
    os.makedirs(model_cache_dir, exist_ok=True)

    model_load_id, use_local_model = resolve_pretrained_model_source(model_name, model_cache_dir)

    # 当本地已有模型时，设置 HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE 环境变量。
    # transformers>=4.45 的 tokenizer __init__ 内部 _patch_mistral_regex 会调用
    # model_info(model_id) 发起 HTTP 请求，仅靠 local_files_only=True 无法阻止。
    _prev_hub_offline = os.environ.get("HF_HUB_OFFLINE")
    _prev_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    if use_local_model:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # DDP 环境下让 Rank 0 先下载
    if ddp_enabled and world_size > 1 and rank != 0:
        dist.barrier()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_load_id,
            cache_dir=model_cache_dir,
            local_files_only=use_local_model,
        )
    finally:
        # 恢复环境变量，避免影响后续需要网络的操作（如数据集下载）
        if _prev_hub_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = _prev_hub_offline
        if _prev_tf_offline is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = _prev_tf_offline

    if task_type == "nlu":
        # 如果是 GLUE 任务，load_dataset("glue", task_name)
        use_local_ds = check_local_dataset("glue", task_name, dataset_cache_dir)
        if use_local_ds:
            os.environ["HF_DATASETS_OFFLINE"] = "1"
        dataset = load_dataset(
            "glue", 
            task_name, 
            cache_dir=dataset_cache_dir
        )
        if use_local_ds:
            os.environ.pop("HF_DATASETS_OFFLINE", None)

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

        train_split, val_split = glue_nlu_train_val_splits(dataset, task_name)
        _assert_glue_split_has_gold_labels(dataset, train_split, task_name)
        if val_split != train_split:
            _assert_glue_split_has_gold_labels(dataset, val_split, task_name)

        tokenized = dataset.map(tokenize_fn, batched=True)
        tokenized = tokenized.rename_column("label", "labels")
        keep_cols = ["input_ids", "attention_mask", "labels"]
        ref_cols = tokenized[train_split].column_names
        if "token_type_ids" in ref_cols:
            keep_cols.append("token_type_ids")
        tokenized = tokenized.remove_columns([c for c in ref_cols if c not in keep_cols])
        tokenized.set_format(type="torch")

        collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

        def _make_loaders_nlu(seed: int):
            """根据给定 seed 重建 NLU DataLoader。
            多 seed 实验中每个 seed 独立调用，确保 DistributedSampler 采样随机性相互独立。
            """
            _val_loader_eval_full: Optional[Union[DataLoader, Dict[str, DataLoader]]] = None
            if ddp_enabled and world_size > 1:
                _train_sampler = DistributedSampler(
                    tokenized[train_split],
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=seed,  # 每个 seed 独立，避免多 seed 实验共享同一 shuffle 顺序
                    drop_last=False,
                )
                # 训练内 mini-val（EvoRank trial）保持每卡步数一致，使用 drop_last=True。
                _val_sampler = DistributedSampler(
                    tokenized[val_split],
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    seed=seed,
                    drop_last=True,
                )
                # 评估指标使用不丢样本的分片验证集，后续 all_gather 聚合。
                _val_eval_sampler = _DistributedEvalSampler(
                    tokenized[val_split],
                    num_replicas=world_size,
                    rank=rank,
                )
                _train_loader = DataLoader(
                    tokenized[train_split], batch_size=batch_size, sampler=_train_sampler, collate_fn=collator,
                    num_workers=dataloader_num_workers, pin_memory=pin_memory
                )
                _val_loader = DataLoader(
                    tokenized[val_split], batch_size=batch_size, sampler=_val_sampler, collate_fn=collator,
                    num_workers=dataloader_num_workers, pin_memory=pin_memory
                )
                _val_loader_eval = DataLoader(
                    tokenized[val_split], batch_size=batch_size, sampler=_val_eval_sampler, collate_fn=collator,
                    num_workers=dataloader_num_workers, pin_memory=pin_memory
                )
                if task_name == "mnli":
                    _val_m_sampler = _DistributedEvalSampler(
                        tokenized["validation_matched"],
                        num_replicas=world_size,
                        rank=rank,
                    )
                    _val_mm_sampler = _DistributedEvalSampler(
                        tokenized["validation_mismatched"],
                        num_replicas=world_size,
                        rank=rank,
                    )
                    _val_m_loader = DataLoader(
                        tokenized["validation_matched"],
                        batch_size=batch_size,
                        sampler=_val_m_sampler,
                        collate_fn=collator,
                        num_workers=dataloader_num_workers, pin_memory=pin_memory
                    )
                    _val_mm_loader = DataLoader(
                        tokenized["validation_mismatched"],
                        batch_size=batch_size,
                        sampler=_val_mm_sampler,
                        collate_fn=collator,
                        num_workers=dataloader_num_workers, pin_memory=pin_memory
                    )
                    _val_loader_eval_full = {"matched": _val_m_loader, "mismatched": _val_mm_loader}
                else:
                    _val_loader_eval_full = _val_loader_eval
            else:
                _train_loader = DataLoader(tokenized[train_split], batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=dataloader_num_workers, pin_memory=pin_memory)
                _val_loader = DataLoader(tokenized[val_split], batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=dataloader_num_workers, pin_memory=pin_memory)
                if task_name == "mnli":
                    _val_m_loader = DataLoader(tokenized["validation_matched"], batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=dataloader_num_workers, pin_memory=pin_memory)
                    _val_mm_loader = DataLoader(tokenized["validation_mismatched"], batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=dataloader_num_workers, pin_memory=pin_memory)
                    _val_loader_eval_full = {"matched": _val_m_loader, "mismatched": _val_mm_loader}
                else:
                    _val_loader_eval_full = _val_loader
            return _train_loader, _val_loader, _val_loader_eval_full

        train_loader, val_loader, val_loader_eval_full = _make_loaders_nlu(seed)

        if task_name == "stsb":
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_load_id,
                num_labels=1,
                problem_type="regression",
                cache_dir=model_cache_dir,
                local_files_only=use_local_model,
            )
        else:
            label_feature = dataset[train_split].features.get("label", None)
            if label_feature is not None and hasattr(label_feature, "num_classes") and label_feature.num_classes is not None:
                num_labels = int(label_feature.num_classes)
            else:
                num_labels = len(set(dataset[train_split]["label"]))
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_load_id,
                num_labels=num_labels,
                cache_dir=model_cache_dir,
                local_files_only=use_local_model,
            )
        
        # Rank 0 完成加载/下载后释放其它进程
        if ddp_enabled and world_size > 1 and rank == 0:
            dist.barrier()
        if rank == 0 and task_name in {"cola", "rte"}:
            train_labels = dataset[train_split]["label"]
            val_labels = dataset[val_split]["label"]
            train_dist = {str(int(k)): int(sum(1 for x in train_labels if int(x) == int(k))) for k in sorted(set(train_labels))}
            val_dist = {str(int(k)): int(sum(1 for x in val_labels if int(x) == int(k))) for k in sorted(set(val_labels))}
            # region agent log
            _debug_log(
                run_id=f"{task_name}-setup-seed{seed}",
                hypothesis_id="H2",
                location="run_benchmark.py:setup_data_and_model",
                message="glue_label_distribution",
                data={
                    "task": task_name,
                    "train_split": train_split,
                    "val_split": val_split,
                    "train_size": len(train_labels),
                    "val_size": len(val_labels),
                    "train_label_dist": train_dist,
                    "val_label_dist": val_dist,
                    "num_labels": int(getattr(base_model.config, "num_labels", -1)),
                },
            )
            # endregion
        return train_loader, val_loader, val_loader_eval_full, base_model, tokenizer, _make_loaders_nlu

    if task_type == "nlg":
        if nlg_dataset_name == "cnn_dailymail":
            use_local_ds = check_local_dataset("cnn_dailymail", "3.0.0", dataset_cache_dir)
            if use_local_ds:
                os.environ["HF_DATASETS_OFFLINE"] = "1"
            dataset = load_dataset(
                "cnn_dailymail", 
                "3.0.0", 
                cache_dir=dataset_cache_dir
            )
            if use_local_ds:
                os.environ.pop("HF_DATASETS_OFFLINE", None)
            text_key = "article"
            target_key = "highlights"
        elif nlg_dataset_name == "xsum":
            use_local_ds = check_local_dataset("xsum", None, dataset_cache_dir)
            if use_local_ds:
                os.environ["HF_DATASETS_OFFLINE"] = "1"
            dataset = load_dataset(
                "xsum", 
                cache_dir=dataset_cache_dir
            )
            if use_local_ds:
                os.environ.pop("HF_DATASETS_OFFLINE", None)
            text_key = "document"
            target_key = "summary"
        else:
            raise NotImplementedError(f"不支持的 nlg_dataset_name: {nlg_dataset_name!r}。已支持: cnn_dailymail, xsum")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        def preprocess(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
            inputs = examples[text_key]
            targets = examples[target_key]
            model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
            labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
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

        def _make_loaders_nlg(seed: int):
            """根据给定 seed 重建 NLG DataLoader。
            多 seed 实验中每个 seed 独立调用，确保 DistributedSampler 采样随机性相互独立。
            """
            if ddp_enabled and world_size > 1:
                _train_sampler = DistributedSampler(
                    tokenized["train"],
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=seed,  # 每个 seed 独立，避免多 seed 实验共享同一 shuffle 顺序
                    drop_last=False,
                )
                _val_sampler = DistributedSampler(
                    tokenized[val_split_name],
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    seed=seed,
                    drop_last=True,
                )
                _val_eval_sampler = _DistributedEvalSampler(
                    tokenized[val_split_name],
                    num_replicas=world_size,
                    rank=rank,
                )
                _train_loader = DataLoader(
                    tokenized["train"], batch_size=batch_size, sampler=_train_sampler, collate_fn=collator,
                    num_workers=dataloader_num_workers, pin_memory=pin_memory
                )
                _val_loader = DataLoader(
                    tokenized[val_split_name], batch_size=batch_size, sampler=_val_sampler, collate_fn=collator,
                    num_workers=dataloader_num_workers, pin_memory=pin_memory
                )
                _val_loader_eval_full = DataLoader(
                    tokenized[val_split_name], batch_size=batch_size, sampler=_val_eval_sampler, collate_fn=collator,
                    num_workers=dataloader_num_workers, pin_memory=pin_memory
                )
            else:
                _train_loader = DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=dataloader_num_workers, pin_memory=pin_memory)
                _val_loader = DataLoader(tokenized[val_split_name], batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=dataloader_num_workers, pin_memory=pin_memory)
                _val_loader_eval_full = None
            return _train_loader, _val_loader, _val_loader_eval_full

        train_loader, val_loader, val_loader_eval_full = _make_loaders_nlg(seed)

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_load_id,
            cache_dir=model_cache_dir,
            local_files_only=use_local_model,
        )

        # Rank 0 完成加载/下载后释放其它进程
        if ddp_enabled and world_size > 1 and rank == 0:
            dist.barrier()
            
        return train_loader, val_loader, val_loader_eval_full, base_model, tokenizer, _make_loaders_nlg

    raise ValueError(f"未知 task_type: {task_type}")


def peft_factory(
    model: nn.Module,
    method_name: str,
    target_rank: int = 8,
    total_steps: Optional[int] = None,
    adalora_delta_t: int = 200,
    train_loader: Optional[DataLoader] = None,
    task_type: str = "nlu",
    is_main_process: bool = True,
    ddp_enabled: bool = False,
    adalora_init_r: Optional[int] = None,
    adalora_tinit: Optional[int] = None,
    adalora_tfinal: Optional[int] = None,
    adalora_orth_reg_weight: float = 0.1,
    nlu_regression: bool = False,
    lora_alpha: Optional[float] = None,
    target_modules_override: Optional[str] = None,
    module_preset: str = "default",
    comparison_protocol: str = "none",
    protocol_dropout: float = 0.05,
    evorank_r_max: int = 16,
    evorank_use_rslora: bool = False,
    evorank_alpha_u: float = 1.0,
    evorank_beta_u: float = 1.0,
    evorank_rho: float = 0.9,
    evorank_p_g: float = 0.8,
    evorank_p_p: float = 0.1,
    evorank_H_g: int = 2,
    evorank_H_p: int = 3,
    evorank_cooldown_steps: int = 2,
    evorank_allow_reallocation: bool = True,
    evorank_max_reallocate_candidates: int = 8,
    evorank_compensation_mode: str = "B",
    toplora_dropout: float = 0.05,
    toplora_lambda_clamp: float =  3.0,
) -> Tuple[nn.Module, Optional[RankEvolutionController], Dict[str, Any]]:
    def _collect_all_linear_target_modules(m: nn.Module) -> List[str]:
        excluded_fragments = (
            "lm_head",
            "classifier",
            "score",
            "pooler",
            "embed",
            "embedding",
        )
        names: List[str] = []
        for full_name, module in m.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            lname = full_name.lower()
            if any(tok in lname for tok in excluded_fragments):
                continue
            names.append(full_name)
        if not names:
            raise ValueError("未找到可用于 all-linear 注入的 Linear 模块")
        return names

    def _default_target_modules(method: str, model_type_name: str) -> List[str]:
        # 公平对比：所有方法使用相同的 target_modules，确保可训练参数量一致。
        # DeBERTa 6 类模块与 AdaLoRA/SoRA 官方对齐：Q,K,V + attention output + FFN×2
        if "deberta" in model_type_name:
            return ["query_proj", "key_proj", "value_proj", "attention.output.dense", "intermediate.dense", "output.dense"]
        if "roberta" in model_type_name or "bert" in model_type_name:
            return ["query", "key", "value", "intermediate.dense", "output.dense"]
        if "bart" in model_type_name:
            return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        if "llama" in model_type_name or "mistral" in model_type_name:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if "t5" in model_type_name:
            return ["q", "k", "v", "o", "wi", "wo"]
        return ["query", "key", "value", "intermediate.dense", "output.dense"]

    model_type = getattr(getattr(model, "config", None), "model_type", "").lower()

    # --target_modules 优先；否则按 module_preset / 模型类型自动选择默认协议
    if target_modules_override:
        target_modules = [m.strip() for m in target_modules_override.split(",") if m.strip()]
    elif module_preset == "all_linear":
        target_modules = _collect_all_linear_target_modules(model)
    elif module_preset == "attn_only":
        if "deberta" in model_type:
            target_modules = ["query_proj", "key_proj", "value_proj"]
        elif "roberta" in model_type or "bert" in model_type:
            target_modules = ["query", "key", "value"]
        elif "bart" in model_type:
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        elif "llama" in model_type or "mistral" in model_type:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "t5" in model_type:
            target_modules = ["q", "k", "v", "o"]
        else:
            target_modules = ["query", "key", "value"]
    else:
        target_modules = _default_target_modules(method_name, model_type)

    # --lora_alpha 优先；否则回退到 2 * target_rank
    effective_alpha = float(lora_alpha) if lora_alpha is not None else float(2 * target_rank)

    # 公平对比：所有方法统一 target_modules，不再对 LoRA 单独移除 FFN 层。

    # PEFT 的 task_type 需要与 backbone 类型匹配，避免 seq2seq/causal 分支内部逻辑错误。
    if "t5" in model_type:
        peft_task_type = TaskType.SEQ_2_SEQ_LM
    elif "llama" in model_type or "mistral" in model_type:
        peft_task_type = TaskType.CAUSAL_LM
    else:
        peft_task_type = TaskType.SEQ_CLS

    controller: Optional[RankEvolutionController] = None
    effective_dropout_val: Optional[float] = None

    if task_type == "nlu":
        modules_to_save = ["classifier", "score", "pooler"]
        # DeBERTa 上 pooler 经常保持在 backbone 原 dtype（常见为 fp16/bf16）执行。
        # 将其作为 modules_to_save 训练并与 LoRA/头部参数混合后，容易出现 dtype 不一致路径；
        # 对 GLUE 任务仅保留 classifier/score 可避免该问题。
        if "deberta" in model_type:
            modules_to_save = ["classifier", "score"]
    else:
        modules_to_save = None

    if method_name in ("lora", "flatlora", "pissa"):
        # DeBERTa：LoRA 梯度幅度随层内激活范数放大，易出现早期数值爆炸（见 huggingface/peft#3073 等讨论）。
        # RSLoRA（alpha/sqrt(r)）可显著缓和该问题；与 AdaLoRA/其他方法对比时仅影响标准 lora 分支。
        # PiSSA：PEFT 原生 SVD 主成分初始化 + 残差基座冻结，训练循环与标准 LoRA 相同（仅对 pissa 设置 init_lora_weights）。
        # PiSSA 要求 dropout=0：主成分奇异向量被 dropout 随机丢弃会破坏 SVD 初始化优势（见 PiSSA/README.md）。
        if comparison_protocol == "controlled_fair":
            _lora_dropout = float(protocol_dropout)
        else:
            _lora_dropout = 0.0 if "deberta" in model_type else 0.1
        # PiSSA 必须固定 dropout=0，不受统一 protocol_dropout 覆盖
        if method_name == "pissa":
            _lora_dropout = 0.0
        effective_dropout_val = float(_lora_dropout)
        # PiSSA 论文要求 lora_alpha == r（scaling=1）以保证 SVD 初始化的有效权重 == 预训练权重。
        # alpha != r 时虽然 PEFT 会在 SVD 分解中补偿初始值，但 scaling 仍会放大训练梯度，
        # 导致 PiSSA 的非零 A/B 初始化在小数据集上剧烈震荡甚至崩溃（CoLA/RTE 实测 MCC=0）。
        _pissa_alpha = effective_alpha
        if method_name == "pissa":
            _pissa_alpha = float(target_rank)
            if is_main_process and abs(_pissa_alpha - effective_alpha) > 0.01:
                print(
                    f"[pissa] lora_alpha 自动覆盖: {effective_alpha} -> {_pissa_alpha} "
                    f"(PiSSA 要求 alpha==r={target_rank} 保证 scaling=1)"
                )
        _lora_kw: Dict[str, Any] = dict(
            task_type=peft_task_type,
            r=target_rank,
            lora_alpha=_pissa_alpha if method_name == "pissa" else effective_alpha,
            lora_dropout=_lora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            bias="none",
            # RSLoRA 与极小 warmup lr 下的 Adam 首步曾在日志中与 classifier 范数发散同时出现；标准缩放先保证稳定。
            use_rslora=False,
        )
        if method_name == "pissa":
            init_method = getattr(args, "pissa_init_method", "pissa") if args else "pissa"
            _lora_kw["init_lora_weights"] = init_method
        config = LoraConfig(**_lora_kw)
        model = get_peft_model(model, config)

    elif method_name == "adalora":
        # AdaLoRA：预算调度 + RankAllocator（步后 update_and_allocate）+ 正交正则（本脚本在 loss 上显式加入，见 run_training_loop）。
        planned_steps = max(int(total_steps or 1000), 1)
        tinit, tfinal, sched_warn = normalize_adalora_schedule(
            total_steps=planned_steps,
            adalora_tinit=adalora_tinit,
            adalora_tfinal=adalora_tfinal,
        )
        if sched_warn is not None and is_main_process:
            print(sched_warn)
        init_r_val = int(adalora_init_r) if adalora_init_r is not None else target_rank * 2
        if init_r_val < target_rank:
            raise ValueError("adalora_init_r 应 >= target_rank（target_r）")

        # 与 LoRA/PiSSA 保持一致：DeBERTa 上 dropout=0.0，其他模型 dropout=0.1
        _adalora_dropout = 0.0 if "deberta" in model_type else 0.1
        if comparison_protocol == "controlled_fair":
            _adalora_dropout = float(protocol_dropout)
        effective_dropout_val = float(_adalora_dropout)
        adalora_kw: Dict[str, Any] = dict(
            task_type=peft_task_type,
            init_r=init_r_val,
            target_r=target_rank,
            lora_alpha=effective_alpha,
            lora_dropout=_adalora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
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
            tgt_orth = float(adalora_orth_reg_weight)
            # 原版 AdaLoRA 论文在小数据集 (CoLA/RTE) 上使用 reg_orth_coef=0.1~0.3，
            # 正交正则对 SVD 秩稳定性至关重要，不应因数据集小而强制关闭。
            # 仅兜底 PEFT 底层 assert orth_reg_weight > 0 的硬性要求。
            if tgt_orth <= 0:
                print("Warning: PEFT AdaLoRA requires orth_reg_weight > 0. Clamping to 1e-8 to avoid crash.")
                tgt_orth = 1e-8
                
            adalora_kw["orth_reg_weight"] = tgt_orth
        config = AdaLoraConfig(**adalora_kw)
        model = get_peft_model(model, config)

    elif method_name == "evorank":
        er_max = int(evorank_r_max)
        controller = inject_evo_lora(
            model=model,
            target_modules=target_modules,
            layer_kwargs={
                "r_max": er_max,
                "r_init": target_rank,
                "lora_alpha": effective_alpha,
                "use_rslora": bool(evorank_use_rslora),
                "compensation_mode": evorank_compensation_mode, # 补偿模式：B, A, 或 Both
            },
            controller_kwargs={
                "rho": float(evorank_rho),
                "p_g": float(evorank_p_g),
                "p_p": float(evorank_p_p),
                "H_g": int(evorank_H_g),
                "H_p": int(evorank_H_p),
                "cooldown_steps": int(evorank_cooldown_steps),
                "r_min": 2,
                "r_max": er_max,
                "alpha_u": float(evorank_alpha_u),
                "beta_u": float(evorank_beta_u),
                "allow_reallocation": bool(evorank_allow_reallocation),
                "expand_init_mode": getattr(args, "expand_init_mode", "zero"),
                "max_reallocate_candidates": int(evorank_max_reallocate_candidates) if int(evorank_max_reallocate_candidates) > 0 else None,
            },
        )
        # EvoRank 手动注入后，需要显式解冻任务头（与 HF PEFT 在 SEQ_CLS 下的行为对齐）。
        # DeBERTa 上与标准 LoRA 一致：不训练 pooler，避免 dtype/数值路径问题。
        for name, param in model.named_parameters():
            if "classifier" in name or "score" in name or "lm_head" in name or name == "shared":
                param.requires_grad = True
            elif "pooler" in name and "deberta" not in model_type:
                param.requires_grad = True

    elif method_name == "sora":
        # 与 LoRA/PiSSA 保持一致：DeBERTa 上 dropout=0.0，其他模型 dropout=0.1
        _sora_dropout = 0.0 if "deberta" in model_type else 0.1
        if comparison_protocol == "controlled_fair":
            _sora_dropout = float(protocol_dropout)
        effective_dropout_val = float(_sora_dropout)
        inject_sora(
            model=model,
            target_modules=target_modules,
            r=target_rank,
            lora_alpha=effective_alpha,
            lora_dropout=_sora_dropout,
        )

    elif method_name == "toplora":
        # TopLoRA (NeurIPS 2025): token-wise singular value scaling on LoRA
        # 与 SoRA 同构注入模式，秩固定，不支持 merge
        _toplora_dropout = float(toplora_dropout)
        if comparison_protocol == "controlled_fair":
            _toplora_dropout = float(protocol_dropout)
        inject_toplora(
            model=model,
            target_modules=target_modules,
            r=target_rank,
            lora_alpha=effective_alpha,
            lora_dropout=_toplora_dropout,
            lambda_clamp=toplora_lambda_clamp,
        )
        effective_dropout_val = float(_toplora_dropout)

    else:
        raise ValueError(f"未知 method_name: {method_name}")

    wrapped_model = DictFeatureClassifier(model)

    trainable_params = count_trainable_params(wrapped_model)

    # TopLoRA 每层额外引入 W_λ (d_in × r) 参数，同 rank 下参数量高于 LoRA/PiSSA。
    # 单独统计并写入 meta，便于 CSV 中披露容量差异（公平对比必须项）。
    extra_params = 0
    if method_name == "toplora":
        for _, m in wrapped_model.named_modules():
            if type(m).__name__ == "TopLoRALinear":
                extra_params += m.lora_lambda.weight.numel()

    meta = {
        "trainable_params": trainable_params,
        "extra_params": extra_params,  # TopLoRA 专属额外参数量（其他方法为 0）
        "target_modules": list(target_modules),
        "effective_dropout": effective_dropout_val,
    }
    return wrapped_model, controller, meta


def _adalora_post_step_update(model: nn.Module, global_step: int) -> None:
    """PEFT AdaLora 步后 RankAllocator / mask 更新（兼容 DDP + DictFeatureClassifier）。"""
    adalora_update_and_allocate(model, global_step)


def _collect_rank_distribution(
    model: nn.Module,
    method_name: str,
    controller: Optional["RankEvolutionController"] = None,
    target_rank: int = 8,
) -> Dict[str, Dict[str, int]]:
    """
    收集每一层 LoRA 注入层的有效秩信息。

    返回:
        {
            "per_layer": {layer_name: effective_rank, ...},
            "summary": {"avg_rank": float, "total_active": int, "total_capacity": int},
        }
    """
    per_layer: Dict[str, int] = {}
    inner = _unwrap_for_save(model)

    if method_name == "evorank" and controller is not None:
        for name, layer in controller.layers.items():
            per_layer[name] = layer.get_active_rank()
        total_capacity = sum(layer.r_max for layer in controller.layers.values())

    elif method_name == "adalora":
        extra_info = ""
        eff_source = "lora_E"
        rank_pattern_capacity = 0
        try:
            for mod in inner.modules():
                if hasattr(mod, "rankallocator"):
                    rank_alloc = mod.rankallocator
                    # _current_threshold 存在于 rankallocator 中 (根据 peft 源码)
                    if hasattr(rank_alloc, "threshold"):
                        extra_info = f"threshold={rank_alloc.threshold:.4f}"
                    break
        except Exception:
            pass

        # AdaLoRA: 优先使用 peft_config.rank_pattern（更贴近 PEFT 内部的“有效秩”口径）。
        # 若不可用，再回退到 lora_E 非零计数（在极短步数下可能全 0，容易误读）。
        try:
            cfg = getattr(inner, "peft_config", {}).get("default", None)
            rank_pattern = getattr(cfg, "rank_pattern", None) if cfg is not None else None
            parsed_from_rank_pattern: Dict[str, int] = {}
            if isinstance(rank_pattern, dict) and len(rank_pattern) > 0:
                for k, v in rank_pattern.items():
                    try:
                        if isinstance(v, (list, tuple)):
                            parsed_from_rank_pattern[str(k)] = int(sum(bool(x) for x in v))
                            rank_pattern_capacity += int(len(v))
                        elif torch.is_tensor(v):
                            parsed_from_rank_pattern[str(k)] = int(v.view(-1).sum().item())
                            rank_pattern_capacity += int(v.numel())
                        else:
                            parsed_from_rank_pattern[str(k)] = int(v)
                            rank_pattern_capacity += max(int(target_rank), int(v))
                    except Exception:
                        continue
            if parsed_from_rank_pattern:
                per_layer.update(parsed_from_rank_pattern)
                eff_source = "rank_pattern"
        except Exception:
            pass

        if not per_layer:
            for n, p in inner.named_parameters():
                if "lora_E" in n and p.numel() > 0:
                    eff_r = int((p.data.abs() > 1e-9).sum().item())
                    # 从参数名提取层路径（去掉 .lora_E.default.weight 等后缀）
                    layer_key = n.replace(".lora_E.default.weight", "").replace(".lora_E.weight", "").replace(".lora_E", "")
                    per_layer[layer_key] = eff_r

        if eff_source == "rank_pattern":
            total_capacity = rank_pattern_capacity if rank_pattern_capacity > 0 else (len(per_layer) * max(int(target_rank), 1))
        else:
            total_capacity = sum(p.numel() for n, p in inner.named_parameters() if "lora_E" in n)

    elif method_name == "sora":
        # SoRA: gate 中非零元素个数 = 有效秩
        # 获取 optimizer 中正在使用的 lambda_2 阈值，如果能拿到的话
        extra_info = ""
        # 简化处理：这只是一个提示，我们没有直接传递当前 lambda，只能查参数
        for n, p in inner.named_parameters():
            if n.endswith(".gate"):
                active_gates = int((p.data.abs() > 1e-9).sum().item())
                layer_key = n.replace(".gate", "")
                per_layer[layer_key] = active_gates
        total_capacity = sum(
            p.numel() for n, p in inner.named_parameters() if n.endswith(".gate")
        )
        gate_nan_count = int(sum(
            torch.isnan(p.detach()).sum().item()
            for n, p in inner.named_parameters()
            if n.endswith(".gate")
        ))
        gate_abs_vals = torch.cat(
            [p.detach().abs().view(-1) for n, p in inner.named_parameters() if n.endswith(".gate") and p.numel() > 0]
        ) if total_capacity > 0 else None
        if gate_abs_vals is not None and gate_abs_vals.numel() > 0:
            summary_gate_stats = {
                "min": float(gate_abs_vals.min().item()),
                "max": float(gate_abs_vals.max().item()),
                "mean": float(gate_abs_vals.mean().item()),
            }
        else:
            summary_gate_stats = None

    else:
        # LoRA / TopLoRA: 固定秩
        for n, p in inner.named_parameters():
            if "lora_A" in n and p.numel() > 0:
                layer_key = n.replace(".lora_A.default.weight", "").replace(".lora_A.weight", "").replace(".lora_A", "")
                per_layer[layer_key] = target_rank
        total_capacity = len(per_layer) * target_rank

    total_active = sum(per_layer.values())
    n_layers = max(len(per_layer), 1)
    avg_rank = total_active / n_layers

    base_rank = target_rank
    if method_name == "adalora":
        config = getattr(inner, "peft_config", {}).get("default", None)
        if config is not None:
            base_rank = getattr(config, "init_r", target_rank * 2)

    summary = {
        "avg_rank": round(avg_rank, 4),
        "total_active": total_active,
        "total_capacity": total_capacity,
        "base_rank": base_rank,
    }
    if method_name == "adalora":
        summary["eff_source"] = eff_source
        cfg = getattr(inner, "peft_config", {}).get("default", None)
        if cfg is not None:
            summary["adalora_config_diag"] = {
                "init_r": int(getattr(cfg, "init_r", base_rank)),
                "target_r": int(getattr(cfg, "target_r", target_rank)),
                "total_step": int(getattr(cfg, "total_step", -1)),
                "tinit": int(getattr(cfg, "tinit", -1)),
                "tfinal": int(getattr(cfg, "tfinal", -1)),
                "deltaT": int(getattr(cfg, "deltaT", -1)),
            }
    if method_name == "sora":
        summary["gate_abs_stats"] = summary_gate_stats if "summary_gate_stats" in locals() else None
        summary["gate_nan_count"] = gate_nan_count if "gate_nan_count" in locals() else 0
    if 'extra_info' in locals() and extra_info:
        summary["extra_string"] = extra_info + (f", eff_source={eff_source}" if method_name == "adalora" else "")

    return {
        "per_layer": per_layer,
        "summary": summary,
    }


def _print_rank_distribution(
    rank_info: Dict[str, Any],
    method_name: str,
    epoch: int,
    epochs: int,
) -> None:
    """格式化打印逐层秩分布。"""
    per_layer = rank_info["per_layer"]
    summary = rank_info["summary"]
    
    extra_str = f" [{summary['extra_string']}]" if "extra_string" in summary else ""
    print(f"[{method_name}] === Rank Distribution (epoch={epoch}/{epochs}){extra_str} ===")
    
    if method_name not in ("lora", "toplora", "pissa", "flatlora"):
        base_rank = summary.get("base_rank", -1)
        omitted = 0
        for layer_name, eff_r in per_layer.items():
            if eff_r == base_rank:
                omitted += 1
                continue
            # 简化层名：只保留 layer.X.xxx.yyy 部分
            short_name = layer_name
            parts = layer_name.split(".")
            # 尝试从 'layer' 关键字开始截取
            for i, part in enumerate(parts):
                if part == "layer":
                    short_name = ".".join(parts[i:])
                    break
            label = "eff_r" if method_name == "adalora" else (
                "gates" if method_name == "sora" else "r"
            )
            print(f"  {short_name}: {label}={eff_r}")
        
        if omitted > 0:
            print(f"  ... (+ {omitted} layers unchanged at {base_rank})")
            
    print(
        f"  avg_rank={summary['avg_rank']:.2f}  "
        f"total_active={summary['total_active']}/{summary['total_capacity']}"
    )
    if float(summary.get("avg_rank", 0.0)) == 0.0:
        if method_name == "adalora":
            cfg_diag = summary.get("adalora_config_diag", {})
            if cfg_diag:
                print(
                    "  [diag] adalora_config: "
                    f"init_r={cfg_diag.get('init_r')} "
                    f"target_r={cfg_diag.get('target_r')} "
                    f"total_step={cfg_diag.get('total_step')} "
                    f"tinit={cfg_diag.get('tinit')} "
                    f"tfinal={cfg_diag.get('tfinal')} "
                    f"deltaT={cfg_diag.get('deltaT')}"
                )
            print("  [diag] lora_E 在 PEFT AdaLoRA 中零初始化；极短步数可能尚未形成非零有效秩。")
        elif method_name == "sora":
            gate_stats = summary.get("gate_abs_stats")
            if gate_stats is not None:
                print(
                    "  [diag] gate|abs| stats: "
                    f"min={gate_stats['min']:.3e} "
                    f"max={gate_stats['max']:.3e} "
                    f"mean={gate_stats['mean']:.3e}"
                )
            gate_nan_count = int(summary.get("gate_nan_count", 0))
            if gate_nan_count > 0:
                print(f"  [diag] gate contains NaN values: count={gate_nan_count}")


def _unwrap_training_module(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _unwrap_for_save(model: nn.Module) -> nn.Module:
    """剥离 DDP 后取 DictFeatureClassifier.inner，避免 state_dict key 带 module. 前缀。"""
    m = _unwrap_training_module(model)
    return getattr(m, "inner", m)


def _state_dict_cpu(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def _extract_tunable_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """仅过滤出被微调（requires_grad=True）的权重以及核心相关的 Buffer（如 active_mask）。"""
    ret = {}
    trainable_keys = {n for n, p in module.named_parameters() if p.requires_grad}
    for k, v in module.state_dict().items():
        # EvoRank 的 active_mask 是 buffer，requires_grad 必然是 False，需硬匹配；
        # 为了容错，也把带有 lora_ / pooler / classifier 的相关层强行抓包下来。
        if k in trainable_keys or "active_mask" in k or "lora_" in k or "classifier" in k or "pooler" in k:
            ret[k] = v.detach().cpu()
    return ret


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
        "model": _extract_tunable_state_dict(inner),
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
    if method_name in ["lora", "adalora", "pissa"] and hasattr(inner, "save_pretrained"):
        inner.save_pretrained(final_dir)
    elif method_name == "toplora":
        # TopLoRA 手动注入，仅保存微调权重
        tunable_sd = _extract_tunable_state_dict(inner)
        torch.save(tunable_sd, os.path.join(final_dir, "model_state.pt"))
    else:
        # 手动注入的方法(SORA/EvoRank)虽然基座有 save_pretrained 但会全量保存，这里强制拦截并仅保存过滤后的微调权重至 .pt 文件
        tunable_sd = _extract_tunable_state_dict(inner)
        torch.save(tunable_sd, os.path.join(final_dir, "model_state.pt"))
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
    head_lr: Optional[float] = None,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "linear",
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
    val_loader_eval_full: Optional[Union[DataLoader, Dict[str, DataLoader]]] = None,
    sora_sparse_lambda: float = 1e-3,
    sora_sparse_lambda_2: float = 1e-3,
    sora_lambda_warmup_steps: int = 0,
    sora_lambda_schedule: Optional[str] = None,
    sora_max_lambda: float = 10.0,
    sora_lambda_num: int = 5,
    task_name: Optional[str] = None,
    checkpoint_root: Optional[str] = None,
    save_steps: int = 0,
    save_every_epoch: bool = False,
    save_final_model: bool = True,
    verify_n_samples: int = 2,
    max_grad_norm: Optional[float] = None,
    flatlora_rho: float = 0.05,
    peft_meta: Optional[Dict[str, Any]] = None,
    evo_include_noop_candidate: bool = True,
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
            find_unused_parameters=True,
        )
    # 默认与适配器同 lr；显式提高分类头速率请传 --head_lr（原先 max(lr,5e-4) 易使 pooler/classifier 相对 LoRA 过快更新）。
    head_lr_val = head_lr if head_lr is not None else lr

    if method_name == "sora":
        # 官方 SoRA trainer.py：区分 LayerNorm/bias（无 decay）和其他参数（有 decay），gate 参数独立优化。
        # 对齐原始实现：get_parameter_names(model, [nn.LayerNorm]) 排除 LayerNorm，再排除 bias。
        _no_decay_names = set()
        for n, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                for pn, _ in m.named_parameters():
                    full_name = f"{n}.{pn}" if n else pn
                    _no_decay_names.add(full_name)
        for n, p in model.named_parameters():
            if "bias" in n:
                _no_decay_names.add(n)

        _head_keys = ["classifier", "score", "lm_head", "shared", "pooler"]
        _non_gate_peft_decay = [p for n, p in model.named_parameters()
            if p.requires_grad and not n.endswith(".gate")
            and not any(k in n for k in _head_keys)
            and n not in _no_decay_names]
        _non_gate_peft_no_decay = [p for n, p in model.named_parameters()
            if p.requires_grad and not n.endswith(".gate")
            and not any(k in n for k in _head_keys)
            and n in _no_decay_names]
        _non_gate_head = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith(".gate") and any(k in n for k in _head_keys)]
        _gate = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith(".gate")]

        sora_wd = 0.1 if weight_decay == 0.01 else weight_decay
        dynamic_wd_peft = sora_wd
        _head_wd = sora_wd
        _adam_wd = sora_wd
        _adam_eps = 1e-6

        optimizer = AdamW([
            {"params": _non_gate_peft_decay, "lr": lr, "weight_decay": sora_wd},
            {"params": _non_gate_peft_no_decay, "lr": lr, "weight_decay": 0.0},
            {"params": _non_gate_head, "lr": head_lr_val, "weight_decay": sora_wd},
        ], weight_decay=_adam_wd, eps=_adam_eps)  # DeBERTa fp16 参数需要 eps=1e-6 防止分母下溢
        sparse_optimizer = SparseAdamW(
            _gate,
            lr=lr,
            sparse_lambda=sora_sparse_lambda_2,
            lambda_schedule=sora_lambda_schedule,
            max_lambda=sora_max_lambda,
            lambda_num=sora_lambda_num,
            weight_decay=0.0,
        )
    else:
        _peft_params = [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in ["classifier", "score", "lm_head", "shared", "pooler"])]
        _head_params = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in ["classifier", "score", "lm_head", "shared", "pooler"])]
        
        # 标准 LoRA / AdaLoRA / EvoRank：适配器矩阵通常不做 weight_decay（与常见 PEFT 复现一致），
        # 否则在 GLUE 等小数据、较高 lr 下易出现数值爆炸（logits/loss NaN）。
        # flatlora：与 nblt/Flat-LoRA 官方训练脚本一致，adapter 侧 weight_decay=0（utils.train_text_to_text_model）
        dynamic_wd_peft = 0.0 if method_name in ("lora", "adalora", "evorank", "toplora", "pissa", "flatlora") else weight_decay
        # 分类头必须保留 weight_decay 防止权重爆炸
        # LoRA A/B 的 wd=0 是为了保护基础权重补偿，但分类头没有这个约束。
        _head_wd = weight_decay
        _adam_wd = 0.0 if method_name in ("lora", "adalora", "evorank", "toplora", "pissa", "flatlora") else weight_decay

        # fp16 参数（如 DeBERTa classifier）在 eps=1e-8 时易在首步触发分母下溢并数值爆炸；
        # evorank 同样训练 fp16 头 + 适配器，与 lora/adalora 统一 eps。
        _adam_eps = 1e-6 if method_name in ("lora", "adalora", "evorank", "toplora", "pissa", "flatlora") else 1e-8
        optimizer = AdamW(
            [
                {"params": _peft_params, "lr": lr, "weight_decay": dynamic_wd_peft},
                {"params": _head_params, "lr": head_lr_val, "weight_decay": _head_wd},
            ],
            weight_decay=_adam_wd,
            eps=_adam_eps,
        )
        sparse_optimizer = None
    
    # Calculate warmup steps early so they can be used in debug logging
    total_train_steps = max_train_steps if max_train_steps is not None else epochs * len(train_loader)
    warmup_steps = int(total_train_steps * warmup_ratio)
    steps_per_epoch = len(train_loader)
    # EvoRank warmup cap：LoRA B=0 零初始化 + 长 warmup 的 ΔW ∝ LR² 二次增长阻滞
    # 会导致训练崩溃（日志实证：warmup=374 → 0.47, warmup=18 → 0.82）。
    # 同时 cap LR warmup 和 ES warmup，上限取 10% epoch（≈30 步），保证 B 在第一
    # epoch 内充分增长、ES 统计量可靠。LR 衰减段不变（仍基于 total_train_steps）。
    evo_warmup_cap = max(20, int(0.1 * steps_per_epoch))
    evo_warmup_steps = min(warmup_steps, evo_warmup_cap) if method_name == "evorank" else warmup_steps
    
    if is_main_process and method_name in {"lora", "evorank", "pissa", "adalora", "sora", "toplora"} and task_name in {"cola", "rte"}:
        if method_name == "sora":
            _dbg_peft_params_cnt = int(sum(
                p.numel()
                for n, p in model.named_parameters()
                if p.requires_grad and not n.endswith(".gate") and not any(k in n for k in _head_keys)
            ))
            _dbg_head_params_cnt = int(sum(
                p.numel()
                for n, p in model.named_parameters()
                if p.requires_grad and not n.endswith(".gate") and any(k in n for k in _head_keys)
            ))
        else:
            _dbg_peft_params_cnt = int(sum(p.numel() for p in _peft_params))
            _dbg_head_params_cnt = int(sum(p.numel() for p in _head_params))
        # region agent log
        _debug_log(
            run_id=f"{task_name}-{method_name}-seed{random_seed}-opt",
            hypothesis_id="H1",
            location="run_benchmark.py:run_training_loop",
            message="optimizer_param_groups",
            data={
                "task": task_name,
                "method": method_name,
                "peft_params": _dbg_peft_params_cnt,
                "head_params": _dbg_head_params_cnt,
                "trainable_params_total": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                "lr": float(lr),
                "head_lr": float(head_lr_val),
                "peft_weight_decay": float(dynamic_wd_peft),
                "head_weight_decay": float(_head_wd),
                "adam_constructor_weight_decay": float(_adam_wd),
                "adam_eps": float(_adam_eps) if method_name != "sora" else None,
                "global_weight_decay": float(weight_decay),
                "pooler_trainable_params": int(sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "pooler" in n)),
                "pooler_trainable_dtypes": sorted({str(p.dtype) for n, p in model.named_parameters() if p.requires_grad and "pooler" in n}),
                "peft_target_modules": list(peft_meta.get("target_modules", [])) if peft_meta else [],
                "lr_warmup_steps": int(warmup_steps),
                "es_warmup_steps": int(evo_warmup_steps) if method_name == "evorank" else None,
                "evo_warmup_cap": int(evo_warmup_cap) if method_name == "evorank" else None,
            },
        )
        # endregion
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

    sora_schedule_stage_steps: Optional[int] = None
    if (
        method_name == "sora"
        and sparse_optimizer is not None
        and sora_lambda_schedule is not None
        and int(sora_lambda_num) > 1
    ):
        sora_schedule_stage_steps = max(1, math.ceil(total_train_steps / int(sora_lambda_num)))
    debug_steps = {0, min(99, max(total_train_steps - 1, 0)), max(total_train_steps - 1, 0)}
    # LoRA+GLUE：在已知易发散区间逐步打点，便于确认首个 NaN 出现在哪一步（不仅依赖硬编码的 99）。
    lora_glue_step_trace = method_name in {"lora", "evorank", "pissa", "flatlora"} and task_name in {"cola", "rte"}
    lora_glue_early_steps = frozenset({1, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 39})
    _lr_warmup = evo_warmup_steps if method_name == "evorank" else warmup_steps
    if lr_scheduler_type == "linear":
        lr_lambda_fn = _linear_warmup_decay_lr_lambda(_lr_warmup, total_train_steps)
    elif lr_scheduler_type == "cosine":
        lr_lambda_fn = _cosine_warmup_decay_lr_lambda(_lr_warmup, total_train_steps)
    else:
        raise ValueError(f"未知 lr_scheduler_type: {lr_scheduler_type}")

    lr_scheduler = LambdaLR(optimizer, lr_lambda_fn, last_epoch=-1)
    if sparse_optimizer is not None:
        sparse_lr_scheduler = LambdaLR(
            sparse_optimizer,
            lr_lambda_fn,
            last_epoch=-1,
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
    
    # === Flat-LoRA PyTorch Hooks 初始化 ===
    flatlora_manager = None
    if method_name == "flatlora" and flatlora_rho > 0:
        from flatlora_inject import FlatLoRAHookManager
        total_steps_for_factor = max_train_steps if max_train_steps is not None else epochs * len(train_loader)
        flatlora_manager = FlatLoRAHookManager(
            model, flatlora_rho, total_steps_for_factor,
            is_main_process=is_main_process, local_rank=local_rank,
        )
    # ==================================
    
    try:
        best_val_acc = 0.0
        best_val_metrics: Dict[str, float] = {}
        train_loss_ema = None
        ema_beta = 0.95
        rouge1_val: float = 0.0
        rouge2_val: float = 0.0
        evorank_es_records: List[Dict[str, float]] = []

        for epoch in range(epochs):
            model.train()
            # DistributedSampler 在每个 epoch 都要 set_epoch，否则会导致采样不同步。
            if ddp_enabled and hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            for batch in train_loader:
                if flatlora_manager is not None:
                    flatlora_manager.prepare_step(global_step)
    
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
                        warmup_steps=evo_warmup_steps,
                        T_es=T_es,
                        lambda_c=lambda_c,
                        complexity_mode=complexity_mode,
                        lambda_pop=lambda_pop,
                        population_strategy=population_strategy,
                        random_seed=random_seed,
                        max_grad_norm=max_grad_norm,
                        include_noop_candidate=evo_include_noop_candidate,
                    )
                    train_loss = float(out["train_loss"])
                    avg_active_rank = float(
                        sum(layer.get_active_rank() for layer in controller.layers.values()) / max(len(controller.layers), 1)
                    )
                    if out.get("did_evolution") and out.get("best_reward") is not None:
                        d_val = out.get("es_delta_val_loss")
                        d_comp = out.get("es_delta_complexity")
                        if d_val is not None and d_comp is not None:
                            evorank_es_records.append(
                                {
                                    "step": float(global_step),
                                    "delta_val_loss": float(d_val),
                                    "delta_complexity": float(d_comp),
                                }
                            )
                            if is_main_process:
                                print(
                                    f"[evorank][es] step={global_step} mutation={out.get('best_mutation')} "
                                    f"delta_val_loss={float(d_val):+.6f} delta_complexity={float(d_comp):+.2f} "
                                    f"base_val_loss={float(out.get('es_base_val_loss')):.6f} "
                                    f"selected_val_loss={float(out.get('es_selected_val_loss')):.6f} "
                                    f"base_complexity={float(out.get('es_base_complexity')):.2f} "
                                    f"selected_complexity={float(out.get('es_selected_complexity')):.2f}"
                                )
                            if writer is not None:
                                writer.add_scalar("evorank/es_delta_val_loss", float(d_val), global_step)
                                writer.add_scalar("evorank/es_delta_complexity", float(d_comp), global_step)
                else:
                    optimizer.zero_grad(set_to_none=True)
                    if sparse_optimizer is not None:
                        sparse_optimizer.zero_grad(set_to_none=True)
                    if flatlora_manager is not None:
                        flatlora_manager.perturb_before_forward()
                    logits = model(features)
                    # 某些环境下 DeBERTa 头部/适配器参数可能为 fp16，直接把 half logits
                    # 喂给损失函数会在反传路径上触发 dtype 冲突（Float vs Half）。
                    # 统一在 loss 入口使用 fp32，可保持数值稳定并避免该类报错。
                    loss = loss_fn(logits.float(), labels)
                    if method_name == "adalora":
                        inner = unwrap_inner_from_training_model(model)
                        ow = get_adalora_orth_reg_weight(inner)
                        if ow > 0:
                            loss = loss + ow * compute_adalora_orthogonal_loss(inner)
                    # --- PiSSA / Flat-LoRA 步级诊断 ---
                    if (
                        is_main_process
                        and method_name in {"pissa", "flatlora"}
                        and (global_step < 5 or global_step % 50 == 0)
                    ):
                        _fl_loss_val = float(loss.detach().item())
                        _fl_logits_mean = float(logits.detach().float().mean().item())
                        _fl_logits_std = float(logits.detach().float().std().item())
                        _fl_finite = bool(torch.isfinite(loss).item())
                        print(
                            f"[{method_name}][step] step={global_step} loss={_fl_loss_val:.4f} "
                            f"finite={_fl_finite} logits_mean={_fl_logits_mean:.4f} logits_std={_fl_logits_std:.4f}"
                        )
                        if not _fl_finite:
                            print(f"[{method_name}][WARNING] loss 不是有限值！训练可能已崩溃。")
                    # --- PiSSA 深度诊断（forward 阶段）：参数范数 & 预测分布 ---
                    _pissa_deep_diag = (
                        is_main_process
                        and method_name == "pissa"
                        and (global_step < 5 or global_step % 100 == 0)
                    )
                    if _pissa_deep_diag:
                        _pred_classes = logits.detach().argmax(dim=-1)
                        _pred_vals, _pred_cnts = _pred_classes.cpu().unique(return_counts=True)
                        _pred_dist_str = " ".join(f"c{int(v)}={int(c)}" for v, c in zip(_pred_vals, _pred_cnts))
                        print(
                            f"[pissa][pred] step={global_step} batch_pred_dist=[{_pred_dist_str}] "
                            f"batch_size={int(logits.size(0))}"
                        )
                    if method_name == "sora":
                        lam = float(sora_sparse_lambda)
                        if sora_lambda_schedule is None and sora_lambda_warmup_steps > 0:
                            lam *= min(1.0, float(global_step + 1) / float(sora_lambda_warmup_steps))
                        # 官方 SoRA：L1 Loss 除以 gate 总元素数做归一化
                        l1_penalty = sum(
                            p.abs().sum() for n, p in model.named_parameters() if n.endswith(".gate")
                        )
                        gate_numel = sum(
                            p.numel() for n, p in model.named_parameters() if n.endswith(".gate")
                        )
                        loss = loss + lam * l1_penalty / max(gate_numel, 1)
                        # --- SoRA 调试输出 ---
                        if is_main_process and (global_step % 50 == 0 or global_step < 5):
                            _gate_vals = torch.cat([p.detach().abs().view(-1) for n, p in model.named_parameters() if n.endswith(".gate")])
                            _gate_zero_ratio = float((_gate_vals < 1e-9).sum().item()) / max(_gate_vals.numel(), 1)
                            _ce_loss = float(loss.detach().item()) - float(lam * l1_penalty.detach().item() / max(gate_numel, 1))
                            print(
                                f"[sora][debug] step={global_step} ce_loss={_ce_loss:.4f} "
                                f"l1_penalty={float(l1_penalty.detach().item()):.4f} "
                                f"lam={lam:.4f} l1_contrib={float(lam * l1_penalty.detach().item() / max(gate_numel, 1)):.6f} "
                                f"gate|abs| min={float(_gate_vals.min().item()):.4e} max={float(_gate_vals.max().item()):.4e} "
                                f"mean={float(_gate_vals.mean().item()):.4e} zero_ratio={_gate_zero_ratio:.4f}"
                            )
                    try:
                        loss.backward()
                    finally:
                        if flatlora_manager is not None:
                            flatlora_manager.restore_after_backward()
                    grad_norm_total: Optional[float] = None
                    # PiSSA 深度诊断（backward 后）：梯度范数 & 参数范数（裁剪前）
                    if _pissa_deep_diag:
                        _a_norms, _b_norms, _a_gnorms, _b_gnorms = [], [], [], []
                        _head_gnorm_pre = None
                        for _n, _p in model.named_parameters():
                            if not _p.requires_grad:
                                continue
                            if "lora_A" in _n:
                                _a_norms.append(float(_p.detach().norm().item()))
                                if _p.grad is not None:
                                    _a_gnorms.append(float(_p.grad.detach().norm().item()))
                            elif "lora_B" in _n:
                                _b_norms.append(float(_p.detach().norm().item()))
                                if _p.grad is not None:
                                    _b_gnorms.append(float(_p.grad.detach().norm().item()))
                            elif _head_gnorm_pre is None and any(
                                k in _n for k in ("classifier", "score", "pooler")
                            ) and _p.grad is not None:
                                _head_gnorm_pre = float(_p.grad.detach().norm().item())
                        _cur_lr = float(optimizer.param_groups[0]["lr"])
                        _a_norm_mean = sum(_a_norms) / max(len(_a_norms), 1)
                        _b_norm_mean = sum(_b_norms) / max(len(_b_norms), 1)
                        _a_gnorm_mean = sum(_a_gnorms) / max(len(_a_gnorms), 1) if _a_gnorms else 0.0
                        _b_gnorm_mean = sum(_b_gnorms) / max(len(_b_gnorms), 1) if _b_gnorms else 0.0
                        _rel_update_a = (_a_gnorm_mean * _cur_lr / max(_a_norm_mean, 1e-12))
                        _rel_update_b = (_b_gnorm_mean * _cur_lr / max(_b_norm_mean, 1e-12))
                        _pissa_grad_norm_pre_clip = float(
                            sum(g**2 for g in (_a_gnorms + _b_gnorms + ([_head_gnorm_pre] if _head_gnorm_pre else [])))**0.5
                        ) if (_a_gnorms or _b_gnorms) else None
                        _head_str = f" head_grad={_head_gnorm_pre:.4f}" if _head_gnorm_pre is not None else ""
                        print(
                            f"[pissa][grad] step={global_step} lr={_cur_lr:.2e} "
                            f"||A||={_a_norm_mean:.4f} ||B||={_b_norm_mean:.4f} "
                            f"||gA||={_a_gnorm_mean:.4f} ||gB||={_b_gnorm_mean:.4f} "
                            f"rel_upd_A={_rel_update_a:.2e} rel_upd_B={_rel_update_b:.2e}"
                            f"{_head_str}"
                        )
                    else:
                        _pissa_grad_norm_pre_clip = None
                    if max_grad_norm is not None and max_grad_norm > 0:
                        _gn = torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], max_grad_norm
                        )
                        grad_norm_total = float(_gn.detach().item()) if isinstance(_gn, torch.Tensor) else float(_gn)
                    if _pissa_deep_diag:
                        _total_gnorm = _pissa_grad_norm_pre_clip if _pissa_grad_norm_pre_clip else (grad_norm_total or 0.0)
                        _clip_ratio = (_total_gnorm / max_grad_norm) if max_grad_norm and max_grad_norm > 0 and _total_gnorm else 0.0
                        print(
                            f"[pissa][clip] step={global_step} "
                            f"grad_norm_pre_clip={_total_gnorm:.4f} "
                            f"max_grad_norm={max_grad_norm} "
                            f"clip_ratio={_clip_ratio:.2f}x "
                            f"{'CLIPPED' if _clip_ratio > 1.0 else 'ok'}"
                        )
                    if (
                        is_main_process
                        and method_name in {"lora", "evorank", "pissa", "flatlora"}
                        and task_name in {"cola", "rte"}
                        and (
                            global_step in debug_steps
                            or (lora_glue_step_trace and 40 <= global_step < 100)
                            or (lora_glue_step_trace and global_step in lora_glue_early_steps)
                        )
                    ):
                        head_norm = None
                        head_grad_norm = None
                        head_dtype = None
                        lora_norm = None
                        lora_grad_norm = None
                        lora_dtype = None
                        head_name = None
                        lora_name = None
                        for n, p in model.named_parameters():
                            if (
                                head_norm is None
                                and p.requires_grad
                                and any(k in n for k in ["classifier", "score", "lm_head", "shared", "pooler"])
                                and n.endswith(".weight")
                            ):
                                head_name = n
                                head_norm = float(p.detach().norm().item())
                                head_dtype = str(p.dtype)
                                if p.grad is not None:
                                    head_grad_norm = float(p.grad.detach().norm().item())
                            if lora_norm is None and p.requires_grad and ("lora_A" in n or "lora_B" in n):
                                lora_name = n
                                lora_norm = float(p.detach().norm().item())
                                lora_dtype = str(p.dtype)
                                if p.grad is not None:
                                    lora_grad_norm = float(p.grad.detach().norm().item())
                            if head_norm is not None and lora_norm is not None:
                                break
                        label_vals, label_counts = labels.detach().cpu().unique(return_counts=True)
                        # region agent log
                        _debug_log(
                            run_id=f"{task_name}-{method_name}-seed{random_seed}-train",
                            hypothesis_id="H1",
                            location="run_benchmark.py:train_step",
                            message="train_step_snapshot",
                            data={
                                "task": task_name,
                                "global_step": int(global_step),
                                "loss": float(loss.detach().item()),
                                "logits_mean": float(logits.detach().float().mean().item()),
                                "logits_std": float(logits.detach().float().std().item()),
                                "batch_label_dist": {str(int(k.item())): int(v.item()) for k, v in zip(label_vals, label_counts)},
                                "head_norm": head_norm,
                                "head_grad_norm": head_grad_norm,
                                "head_dtype": head_dtype,
                                "head_param_name": head_name,
                                "lora_norm": lora_norm,
                                "lora_grad_norm": lora_grad_norm,
                                "lora_dtype": lora_dtype,
                                "lora_param_name": lora_name,
                                "lr_peft": float(optimizer.param_groups[0]["lr"]),
                                "lr_head": float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else None,
                                "optimizer_eps": float(getattr(optimizer, "defaults", {}).get("eps", 1e-8)),
                                "warmup_steps": int(warmup_steps),
                                "grad_norm_total": grad_norm_total,
                            },
                        )
                        # endregion
                    if method_name == "sora":
                        # 防止 gate 梯度中的 NaN/Inf 传入优化器状态，导致后续参数全 NaN。
                        for n, p in model.named_parameters():
                            if n.endswith(".gate") and p.grad is not None and not torch.isfinite(p.grad).all():
                                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    optimizer.step()
                    if (
                        is_main_process
                        and method_name in {"lora", "evorank", "pissa", "flatlora"}
                        and task_name in {"cola", "rte"}
                        and global_step == 0
                    ):
                        # region agent log
                        _hmax = None
                        _hfin: Optional[bool] = None
                        _hdtype: Optional[str] = None
                        for _n, _p in model.named_parameters():
                            if (
                                _p.requires_grad
                                and "classifier" in _n
                                and _n.endswith(".weight")
                            ):
                                _hmax = float(_p.detach().float().abs().max().item())
                                _hfin = bool(torch.isfinite(_p.detach()).all().item())
                                _hdtype = str(_p.dtype)
                                break
                        _debug_log(
                            run_id=f"{task_name}-{method_name}-seed{random_seed}-train",
                            hypothesis_id="H4",
                            location="run_benchmark.py:after_first_optimizer_step",
                            message="post_step0_head_stats",
                            data={
                                "task": task_name,
                                "head_weight_max_abs": _hmax,
                                "head_all_finite": _hfin,
                                "head_dtype": _hdtype,
                                "optimizer_eps": float(getattr(optimizer, "defaults", {}).get("eps", 1e-8)),
                            },
                        )
                        # endregion
                    if sparse_optimizer is not None:
                        sparse_optimizer.step()
                    if method_name == "sora":
                        # 近端更新后再次兜底清洗 gate 参数，避免 NaN 造成 rank 统计全部为 0。
                        for n, p in model.named_parameters():
                            if n.endswith(".gate") and not torch.isfinite(p).all():
                                p.data = torch.nan_to_num(p.data, nan=0.0, posinf=0.0, neginf=0.0)
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
                    method_name == "sora"
                    and sparse_optimizer is not None
                    and sora_schedule_stage_steps is not None
                    and global_step < total_train_steps
                    and global_step % sora_schedule_stage_steps == 0
                ):
                    sparse_optimizer.step_lambda()
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
                if ddp_enabled and _ddp_is_active():
                    dist.barrier()
                val_metric = 0.0
                mkey = glue_primary_metric_key(task_name)
                regression = nlu_is_glue_regression(task_name)
                metrics_dict_val: Dict[str, float] = {}
                y_pred_main: List[float] = []
                y_true_main: List[float] = []
                ev_loader = val_loader_eval_full if val_loader_eval_full is not None else val_loader
                eval_model = _unwrap_training_module(model)
                if isinstance(ev_loader, dict):
                    y_pred_m, y_true_m = _collect_nlu_predictions_distributed(
                        eval_model,
                        ev_loader["matched"],
                        device,
                        regression=regression,
                        ddp_enabled=ddp_enabled,
                        is_main_process=is_main_process,
                    )
                    y_pred_mm, y_true_mm = _collect_nlu_predictions_distributed(
                        eval_model,
                        ev_loader["mismatched"],
                        device,
                        regression=regression,
                        ddp_enabled=ddp_enabled,
                        is_main_process=is_main_process,
                    )
                    if is_main_process:
                        val_metric_m = compute_glue_primary_metric(task_name, y_pred_m, y_true_m)
                        val_metric_mm = compute_glue_primary_metric(task_name, y_pred_mm, y_true_mm)
                        val_metric = (val_metric_m + val_metric_mm) / 2.0
                        metrics_dict_val = compute_glue_metrics_dict(task_name, y_pred_m, y_true_m)
                        metrics_dict_val["accuracy_m"] = val_metric_m
                        metrics_dict_val["accuracy_mm"] = val_metric_mm
                        y_pred_main = y_pred_m
                        y_true_main = y_true_m
                else:
                    y_pred_local, y_true_local = _collect_nlu_predictions_distributed(
                        eval_model,
                        ev_loader,
                        device,
                        regression=regression,
                        ddp_enabled=ddp_enabled,
                        is_main_process=is_main_process,
                    )
                    if is_main_process:
                        val_metric = compute_glue_primary_metric(task_name, y_pred_local, y_true_local)
                        metrics_dict_val = compute_glue_metrics_dict(task_name, y_pred_local, y_true_local)
                        y_pred_main = y_pred_local
                        y_true_main = y_true_local
                if ddp_enabled and _ddp_is_active():
                    m_tensor = torch.tensor([val_metric if is_main_process else 0.0], device=device, dtype=torch.float64)
                    dist.broadcast(m_tensor, src=0)
                    val_metric = float(m_tensor.item())
                    dist.barrier()
                if is_main_process:
                    if val_metric > best_val_acc:
                        best_val_acc = val_metric
                        best_val_metrics = metrics_dict_val
                    elif val_metric == best_val_acc and not best_val_metrics:
                        best_val_metrics = metrics_dict_val
                    if method_name in {"lora", "evorank", "pissa", "flatlora"} and task_name in {"cola", "rte", "mrpc", "stsb"}:
                        pred_list = [int(x) for x in y_pred_main]
                        true_list = [int(x) for x in y_true_main]
                        pred_dist = {str(k): int(sum(1 for x in pred_list if x == k)) for k in sorted(set(pred_list))}
                        true_dist = {str(k): int(sum(1 for x in true_list if x == k)) for k in sorted(set(true_list))}
                        _debug_log(
                            run_id=f"{task_name}-{method_name}-seed{random_seed}-eval",
                            hypothesis_id="H3",
                            location="run_benchmark.py:nlu_eval",
                            message="eval_prediction_distribution",
                            data={
                                "task": task_name,
                                "epoch": int(epoch + 1),
                                "global_step": int(global_step),
                                "metric_key": mkey,
                                "metric_value": float(val_metric),
                                "pred_dist": pred_dist,
                                "true_dist": true_dist,
                                "sample_preds": pred_list[:16],
                                "sample_true": true_list[:16],
                            },
                        )
                        print(
                            f"[{method_name}][eval-dist] epoch={epoch + 1} "
                            f"pred_dist={pred_dist} true_dist={true_dist} "
                            f"sample_pred={pred_list[:8]} sample_true={true_list[:8]}"
                        )
                    print(
                        f"[{method_name}] epoch={epoch + 1}/{epochs} "
                        f"step={global_step} val_{mkey}={val_metric:.4f} best={best_val_acc:.4f}"
                    )
                    # === 逐层秩分布日志 ===
                    rank_info = _collect_rank_distribution(
                        model, method_name, controller=controller,
                        target_rank=int(next(
                            (p.size(0) for n, p in _unwrap_for_save(model).named_parameters()
                             if "lora_A" in n and p.numel() > 0), 8
                        )) if method_name in ("lora", "pissa", "flatlora") else 8,
                    )
                    _print_rank_distribution(rank_info, method_name, epoch + 1, epochs)
                    if writer is not None:
                        writer.add_scalar(f"val/{mkey}", val_metric, epoch)
                        for lname, lr_val in rank_info["per_layer"].items():
                            writer.add_scalar(f"rank/{lname}", lr_val, epoch)
                        writer.add_scalar("rank/avg", rank_info["summary"]["avg_rank"], epoch)
                    if wandb_run is not None:
                        wandb_log_dict = {f"val/{mkey}": val_metric, "epoch": epoch + 1, "step": global_step}
                        wandb_log_dict["rank/avg"] = rank_info["summary"]["avg_rank"]
                        wandb_run.log(wandb_log_dict)
    
            else:
                if tokenizer is None:
                    raise ValueError("task_type='nlg' 时必须传入 tokenizer")
                if ddp_enabled and _ddp_is_active():
                    dist.barrier()
    
                val_metric = 0.0
                rouge1_val = 0.0
                rouge2_val = 0.0
                eval_loader = val_loader_eval_full if val_loader_eval_full is not None else val_loader
                if isinstance(eval_loader, dict):
                    raise ValueError("NLG 评估不支持 dict 类型验证 loader")
    
                # 生成阶段需要底层 seq2seq 模型（可能在 DictFeatureClassifier.inner 或 DDP.module.inner 中）
                if isinstance(model, DDP):
                    inner = getattr(model.module, "inner", model.module)
                    gen_model = inner
                else:
                    gen_model = getattr(model, "inner", model)
    
                preds_text, refs_text = _collect_nlg_text_pairs_distributed(
                    gen_model=gen_model,
                    data_loader=eval_loader,
                    tokenizer=tokenizer,
                    device=device,
                    generation_max_new_tokens=generation_max_new_tokens,
                    nlg_eval_max_samples=nlg_eval_max_samples,
                    ddp_enabled=ddp_enabled,
                    is_main_process=is_main_process,
                )
                if is_main_process:
                    try:
                        import evaluate  # type: ignore
    
                        rouge_metric = evaluate.load("rouge")
                    except Exception as e:
                        # NLG 指标在一些离线/依赖缺失环境下可能无法加载 rouge。
                        # 之前静默写 0 分，导致“全 0”难以定位原因；这里显式打印异常。
                        rouge_metric = None
                        print(f"[warn] evaluate.load('rouge') failed: {type(e).__name__}: {e!r}")
    
                    if rouge_metric is None:
                        if is_main_process:
                            # 避免误以为模型真的完全没有任何重叠
                            print(
                                f"[warn] rouge metric skipped -> 0.0 (preds={len(preds_text)}, refs={len(refs_text)})"
                            )
                        val_metric = 0.0
                        rouge1_val = rouge2_val = 0.0
                    else:
                        if len(preds_text) == 0 or len(refs_text) == 0 or len(preds_text) != len(refs_text):
                            print(
                                f"[warn] rouge inputs look suspicious: preds={len(preds_text)}, refs={len(refs_text)}; "
                                f"pred0={preds_text[0][:80] if preds_text else ''!r}; ref0={refs_text[0][:80] if refs_text else ''!r}"
                            )
                        scores = rouge_metric.compute(predictions=preds_text, references=refs_text)
                        rouge1_val = float(scores.get("rouge1", 0.0))
                        rouge2_val = float(scores.get("rouge2", 0.0))
                        val_metric = float(scores.get("rougeL", scores.get("rougeLsum", 0.0)))
                        if rouge1_val == 0.0 and rouge2_val == 0.0 and val_metric == 0.0:
                            # 如果确实存在可见文本但 rouge 全 0，通常是 tokenize/metric 解析问题或重叠为 0
                            if is_main_process:
                                print(
                                    "[warn] rouge scores all 0.0; "
                                    f"pred0={preds_text[0][:120] if preds_text else ''!r}; "
                                    f"ref0={refs_text[0][:120] if refs_text else ''!r}"
                                )
    
                    best_val_acc = max(best_val_acc, val_metric)
                    print(
                        f"[{method_name}] epoch={epoch + 1}/{epochs} "
                        f"step={global_step} val_rouge1={rouge1_val:.4f} val_rouge2={rouge2_val:.4f} "
                        f"val_rougeL={val_metric:.4f} best={best_val_acc:.4f}"
                    )
                    # === 逐层秩分布日志 (NLG) ===
                    rank_info = _collect_rank_distribution(
                        model, method_name, controller=controller,
                        target_rank=int(next(
                            (p.size(0) for n, p in _unwrap_for_save(model).named_parameters()
                             if "lora_A" in n and p.numel() > 0), 8
                        )) if method_name in ("lora", "pissa", "flatlora") else 8,
                    )
                    _print_rank_distribution(rank_info, method_name, epoch + 1, epochs)
                    if writer is not None:
                        writer.add_scalar("val/rouge1", rouge1_val, epoch)
                        writer.add_scalar("val/rouge2", rouge2_val, epoch)
                        writer.add_scalar("val/rougeL", val_metric, epoch)
                        for lname, lr_val in rank_info["per_layer"].items():
                            writer.add_scalar(f"rank/{lname}", lr_val, epoch)
                        writer.add_scalar("rank/avg", rank_info["summary"]["avg_rank"], epoch)
                    if wandb_run is not None:
                        wandb_log_dict = {
                            "val/rouge1": rouge1_val,
                            "val/rouge2": rouge2_val,
                            "val/rougeL": val_metric,
                            "epoch": epoch + 1,
                            "step": global_step,
                        }
                        wandb_log_dict["rank/avg"] = rank_info["summary"]["avg_rank"]
                        wandb_run.log(wandb_log_dict)
    
                if ddp_enabled and _ddp_is_active():
                    dist.barrier()
    
            if is_main_process and checkpoint_root:
                # 秩分布信息（若上方 eval 尚未计算则此处补算）
                try:
                    _rank_info = rank_info  # type: ignore[possibly-undefined]
                except NameError:
                    _rank_info = _collect_rank_distribution(model, method_name, controller=controller)
                rec: Dict[str, Any] = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "val_metric": float(val_metric),
                    "best_val": float(best_val_acc),
                    "train_loss_ema": float(train_loss_ema) if train_loss_ema is not None else None,
                    "rank_distribution": _rank_info["per_layer"],
                    "rank_summary": _rank_info["summary"],
                }
                if task_type == "nlu" and task_name is not None:
                    if metrics_dict_val:
                        rec.update({k: float(v) for k, v in metrics_dict_val.items()})
                    if task_name == "mnli":
                        rec["glue_metric"] = "m/mm"
                    elif task_name == "qqp":
                        rec["glue_metric"] = "acc/f1"
                    elif task_name == "mrpc":
                        rec["glue_metric"] = "accuracy"
                    else:
                        rec["glue_metric"] = glue_primary_metric_key(task_name)
                if task_type == "nlg":
                    rec["rouge1"] = rouge1_val
                    rec["rouge2"] = rouge2_val
                    rec["rougeL"] = float(val_metric)
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
    finally:
        if flatlora_manager is not None:
            flatlora_manager.remove_hooks()

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

    try:
        rank_dist = _collect_rank_distribution(model, method_name, controller, args.target_rank)
        final_avg_active_rank = rank_dist["summary"]["avg_rank"]
    except Exception:
        final_avg_active_rank = "N/A"

    def _mean_numeric(records: List[Dict[str, float]], key: str) -> Optional[float]:
        vals: List[float] = []
        for rec in records:
            v = rec.get(key)
            if v is None:
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            vals.append(float(v))
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    evorank_es_events = len(evorank_es_records)
    evorank_delta_val_loss_mean = _mean_numeric(evorank_es_records, "delta_val_loss")
    evorank_delta_complexity_mean = _mean_numeric(evorank_es_records, "delta_complexity")

    result = {
        "method": method_name,
        "optimizer_type": "AdamW+SparseAdamW" if method_name == "sora" else "AdamW",
        "total_train_time_sec": total_time,
        "peak_memory_mb": peak_mem_mb,
        "avg_active_rank": final_avg_active_rank,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        "artifact_dir": artifact_dir_str,
        "final_dir": final_dir_str,
        "val_metric_key": val_metric_key_str,
        "evorank_es_events": evorank_es_events if method_name == "evorank" else "",
        "evorank_avg_delta_val_loss": evorank_delta_val_loss_mean if method_name == "evorank" else "",
        "evorank_avg_delta_complexity": evorank_delta_complexity_mean if method_name == "evorank" else "",
        
        "matthews_corrcoef": best_val_metrics.get("matthews_corrcoef", "") if task_type == "nlu" else "",
        "accuracy": best_val_metrics.get("accuracy", "") if task_type == "nlu" else "",
        "accuracy_m": best_val_metrics.get("accuracy_m", "") if task_type == "nlu" else "",
        "accuracy_mm": best_val_metrics.get("accuracy_mm", "") if task_type == "nlu" else "",
        "f1": best_val_metrics.get("f1", "") if task_type == "nlu" else "",
        "pearson_spearman_mean": best_val_metrics.get("pearson_spearman_mean", "") if task_type == "nlu" else "",
        "pearson": best_val_metrics.get("pearson", "") if task_type == "nlu" else "",
        "spearman": best_val_metrics.get("spearman", "") if task_type == "nlu" else "",
        "rouge1": rouge1_val if task_type == "nlg" else "",
        "rouge2": rouge2_val if task_type == "nlg" else "",
        "rougeL": best_val_acc if val_metric_key_str == "rougeL" else "",
    }
    # 与 val_metric_key 对齐的主指标标量，供 CSV 汇总与 Benchmark Summary 读取
    if task_type == "nlu" and val_metric_key_str:
        result[val_metric_key_str] = float(best_val_acc)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EvoRank-LoRA GLUE benchmark runner")
    parser.add_argument(
        "--task_name",
        type=str,
        default="sst2",
        help="NLU：`load_dataset('glue', task_name)` 的子集；含 ax, cola, sst2, …（HF glue/ax 无金标，有监督训练会报错，见 README）。"
        "验证使用各任务 GLUE 官方主指标（见 glue_metrics / README）。NLG：占位/命名用。",
    )
    parser.add_argument("--task_type", type=str, default="nlu", choices=["nlu", "nlg"], help="nlu=GLUE分类，nlg=文本生成")
    parser.add_argument("--nlg_dataset_name", type=str, default="cnn_dailymail", help="nlg 数据集名（如 cnn_dailymail）")
    parser.add_argument("--max_target_length", type=int, default=64, help="nlg 的摘要/目标序列最大长度")
    parser.add_argument("--generation_max_new_tokens", type=int, default=64, help="nlg 生成最大新 tokens")
    parser.add_argument("--nlg_eval_max_samples", type=int, default=200, help="nlg 验证集评测最大样本数（仅用于 ROUGE）")
    parser.add_argument("--dataset_cache_dir", type=str, default="datasets", help="数据集缓存目录（建议仓库内相对路径）")
    parser.add_argument("--model_cache_dir", type=str, default="models", help="模型缓存目录（建议仓库内相对路径）")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["lora", "adalora", "evorank", "sora", "flatlora", "pissa"],
        help="对比方法：含 lora / adalora / evorank / sora / flatlora / toplora / pissa（PEFT PiSSA 初始化）。",
    )
    parser.add_argument("--flatlora_rho", type=float, default=0.05, help="Flat-LoRA 扰动强度参数，对应论文中的 sigma（σ），推荐默认值 0.05")
    parser.add_argument("--target_rank", type=int, default=8)
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=None,
        help="LoRA alpha 缩放系数。None 时默认 2*target_rank。"
             "NLU 对齐 AdaLoRA/SoRA 论文设为 16；NLG（BART+XSum/CNN-DM）对齐 AdaLoRA 论文设为 32。",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default=None,
        help="逗号分隔的注入模块后缀列表，如 'query,key,value,intermediate'。"
             "未指定时按 --module_preset 与模型类型自动选择默认协议。",
    )
    parser.add_argument(
        "--module_preset",
        type=str,
        default="default",
        choices=["default", "attn_only", "all_linear", "custom"],
        help="全局模块预设。default=按模型默认；attn_only=仅注意力投影；all_linear=除分类头外全部线性层；custom=配合 --target_modules 使用。",
    )
    parser.add_argument(
        "--comparison_protocol",
        type=str,
        default="none",
        choices=["none", "controlled_fair", "author_defaults"],
        help="对比协议。controlled_fair=严格控制变量；author_defaults=按论文推荐超参；none=保持当前脚本显式参数。",
    )
    parser.add_argument(
        "--protocol_dropout",
        type=float,
        default=0.05,
        help="comparison_protocol=controlled_fair 时统一的 adapter dropout（推荐 0.05）。",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="DataLoader num_workers 数量，多进程提取加快数据加载")
    parser.add_argument("--pin_memory", action="store_true", help="启用 DataLoader 锁页内存加速")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=None, help="分类头等可训练参数学习率。None 时默认与 --lr 相同；需要更快收敛可显式调高")
    parser.add_argument("--pissa_lr", type=float, default=None, help="PiSSA 专用学习率覆盖。PiSSA 的 SVD 初始化使梯度幅度远大于标准 LoRA，通常需要更低的 lr（论文推荐 1e-4）。None 时使用 --lr")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--T_es", type=int, default=200)
    parser.add_argument("--mini_val_k", type=int, default=8)
    parser.add_argument("--adalora_delta_t", type=int, default=200)
    parser.add_argument("--adalora_init_r", type=int, default=None, help="AdaLoRA init_r，默认 2*target_rank")
    parser.add_argument("--adalora_tinit", type=int, default=None, help="AdaLoRA 初始满秩阶段步数 tinit，默认 floor(0.1*total_step)")
    parser.add_argument(
        "--adalora_tfinal",
        type=int,
        default=None,
        help="AdaLoRA 末尾不调秩阶段步数 tfinal，默认 floor(0.1*total_step)（与 peft_factory 实际默认一致）",
    )
    parser.add_argument(
        "--pissa_init_method",
        type=str,
        default="pissa",
        choices=["pissa", "pissa_niter_16"],
        help="PiSSA SVD 初始化方式。'pissa' 为标准完整 SVD，'pissa_niter_16' 为幂迭代快速SVD（适合大模型和生成任务，大大加快初始化）。",
    )
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
    parser.add_argument(
        "--lambda_c",
        type=float,
        default=0.0,
        help="EvoRank ES 结构奖励中的复杂度惩罚系数：R = -L_val - lambda_c * C(z)。"
             "典型 CE loss 在 0.5~2.0 量级，rank_sum 模式下总秩可达数百，size_aware 模式下可达数十万，"
             "因此建议设为 1e-4 ~ 1e-6 量级。默认 0.0 表示纯 loss 比较（无正则）。",
    )
    parser.add_argument("--complexity_mode", type=str, default="rank_sum", choices=["rank_sum", "size_aware"])
    parser.add_argument(
        "--evorank_r_max",
        type=int,
        default=16,
        help="EvoRank 每层秩超空间上限 R_max（与论文默认 16 一致）",
    )
    parser.add_argument(
        "--evorank_use_rslora",
        action="store_true",
        help="EvoRank 缩放改为 rsLoRA(alpha/sqrt(r))。默认关闭，使用标准 LoRA 缩放(alpha/r)。",
    )
    parser.add_argument(
        "--evo_alpha_u",
        type=float,
        default=1.0,
        help="EvoRank 容量组合 u_ℓ = α·g̃_ℓ + β·s̃̄_ℓ 中 α（见论文式 217）",
    )
    parser.add_argument(
        "--evo_beta_u",
        type=float,
        default=1.0,
        help="EvoRank 容量组合 u_ℓ 中 β",
    )
    parser.add_argument("--evo_rho", type=float, default=0.9, help="EvoRank EMA 平滑系数 rho。设为 0 可消融 EMA 平滑")
    parser.add_argument("--evo_p_g", type=float, default=0.8, help="EvoRank 扩张分位数阈值 p_g")
    parser.add_argument("--evo_p_p", type=float, default=0.1, help="EvoRank 修剪分位数阈值 p_p")
    parser.add_argument("--evo_H_g", type=int, default=2, help="EvoRank 扩张持久计数阈值 H_g。设为 1 可关闭持久化门槛")
    parser.add_argument("--evo_H_p", type=int, default=3, help="EvoRank 修剪持久计数阈值 H_p。设为 1 可关闭持久化门槛")
    parser.add_argument("--evo_cooldown_steps", type=int, default=2, help="EvoRank 扩张后冷却步数。设为 0 可关闭 cooldown")
    parser.add_argument("--evo_allow_reallocation", dest="evo_allow_reallocation", action="store_true", help="EvoRank：启用跨层 reallocation（默认开启）")
    parser.add_argument("--no_evo_allow_reallocation", dest="evo_allow_reallocation", action="store_false", help="EvoRank：关闭跨层 reallocation，只保留 grow / prune")
    parser.set_defaults(evo_allow_reallocation=True)
    parser.add_argument("--evo_include_noop_candidate", dest="evo_include_noop_candidate", action="store_true", help="EvoRank：ES 候选中包含 no-op 基线（默认开启）")
    parser.add_argument("--no_evo_include_noop_candidate", dest="evo_include_noop_candidate", action="store_false", help="EvoRank：移除 no-op 候选，用于消融 validation-side safeguard")
    parser.set_defaults(evo_include_noop_candidate=True)
    parser.add_argument(
        "--expand_init_mode",
        type=str,
        default="zero",
        choices=["zero", "gradient"],
        help="EvoRank 扩张初始化策略。"
             "'zero': B 列清零（安全 cold start）；"
             "'gradient': 论文 Proposition 3.2——基于 ∂L/∂ΔW 的主奇异方向初始化新分量，"
             "通过 power iteration 高效计算，不构造完整梯度矩阵。",
    )
    parser.add_argument(
        "--evo_max_reallocate_candidates",
        type=int,
        default=8,
        help="EvoRank 默认的跨层 reallocation 候选上限。候选按 top-k cross 顺序生成，并在达到该上限后停止。设为 0 或负数可关闭限流（允许组合爆炸消融）。",
    )
    parser.add_argument(
        "--evo_compensation_mode",
        type=str,
        default="B",
        choices=["B", "A", "Both"],
        help="EvoRank 秩变更时的等价变换补偿模式。'B': 只对 B 补偿（默认）；'A': 只对 A 补偿；'Both': A/B 各补偿 sqrt(c)。",
    )
    parser.add_argument("--lambda_pop", type=int, default=None)
    parser.add_argument("--population_strategy", type=str, default="all", choices=["all", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed_list",
        nargs="+",
        type=int,
        default=None,
        help="多种子列表，如 --seed_list 0 21 42 81 100。指定后覆盖 --seed，对每个种子串行运行完整实验。"
             "SoRA 论文使用 5 个种子：0 21 42 81 100。",
    )
    parser.add_argument("--task_list", nargs="+", default=None, help="多任务实验协议，例如: sst2 qnli mnli")
    parser.add_argument("--model_list", nargs="+", default=None, help="多骨干实验协议，例如: roberta-base")
    parser.add_argument("--export_csv", type=str, default="benchmark_results.csv")
    parser.add_argument("--ddp", action="store_true", help="是否启用 torchrun/DistributedDataParallel")
    parser.add_argument("--ddp_backend", type=str, default="nccl", help="DDP backend，通常是 nccl")
    parser.add_argument("--sora_sparse_lambda", type=float, default=1e-3, help="SoRA：gate L1 惩罚系数")
    parser.add_argument("--sora_lambda_warmup_steps", type=int, default=0, help="SoRA：前若干步将 sparse_lambda 从 0 线性升到设定值；0 表示关闭")
    parser.add_argument("--sora_sparse_lambda_2", type=float, default=3e-4, help="SoRA：gate 近端梯度硬裁剪阈值（Proximal Gradient / Soft-Thresholding），对标官方 SparseAdamW")
    parser.add_argument(
        "--sora_lambda_schedule",
        type=str,
        default=None,
        help="SoRA 对 gate 近端阈值 lambda_2 的阶段式调度策略（如 'linear'）。None 表示固定阈值（no-schedule 主线）。",
    )
    parser.add_argument("--sora_max_lambda", type=float, default=10.0, help="SoRA lambda_schedule 的最大 lambda 值")
    parser.add_argument("--sora_lambda_num", type=int, default=5, help="SoRA lambda_schedule 的步数（线性升至 max_lambda 所需步数）")
    parser.add_argument(
        "--toplora_dropout",
        type=float,
        default=0.05,
        help="TopLoRA：dropout 系数（论文默认 0.05）。仅 method=toplora 使用，其他方法忽略。",
    )
    parser.add_argument(
        "--toplora_lr",
        type=float,
        default=None,
        help="TopLoRA 专用学习率覆盖。TopLoRA 的门控 exp(RMSNorm(x@W_λ)) 在高 lr 长训练下容易爆炸"
             "（观测 seed=42 从 epoch=4 起 val=0）；官方脚本 lr=1e-4（Qwen2.5-3B + math_10k）。"
             "None 时若全局 lr>2e-4 自动降为 lr/4（下限 1e-4）。",
    )
    parser.add_argument(
        "--toplora_lambda_clamp",
        type=float,
        default=3.0,
        help="TopLoRA 门控 exp 输入 clamp 上下界 ±C，使 λ∈[e^-C, e^C]，防止 exp 爆炸。"
             "默认 3.0（λ∈[0.05, 20.09]）。设 0 或负数可关闭（对齐官方原始实现，但有数值风险）。",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=None,
        help="梯度裁剪阈值（如 0.1）。None 表示不裁剪。SoRA 论文使用 0.1。",
    )
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
    args = parser.parse_args()
    if args.module_preset == "custom" and not args.target_modules:
        raise ValueError("module_preset=custom 时必须同时提供 --target_modules")
    return args


def run_protocol_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    论文协议批量运行骨架：
    - 支持多任务 × 多骨干 × 多方法 × 多种子
    - 导出 CSV 方便直接填表（主结果/效率/消融）
    """
    import statistics

    tasks = args.task_list if args.task_list else [args.task_name]
    models = args.model_list if args.model_list else [args.model_name]
    seeds = args.seed_list if args.seed_list else [args.seed]
    all_results: List[Dict[str, Any]] = []

    if args.is_main_process and not args.no_output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for task_name in tasks:
        for model_name in models:
            # 数据集 tokenize 在种子循环外（避免重复下载/tokenize），
            # 但 DataLoader（含 DistributedSampler seed）在每个 seed 内重建，
            # 保证多 seed 实验的采样随机性相互独立。
            train_loader, val_loader, val_loader_eval_full, base_model, tokenizer, make_loaders = setup_data_and_model(
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
                seed=seeds[0],
                dataloader_num_workers=args.dataloader_num_workers,
                pin_memory=args.pin_memory,
            )
            planned_total_steps = (
                args.max_train_steps if args.max_train_steps is not None else args.epochs * len(train_loader)
            )

            current_sora_lambda_schedule = args.sora_lambda_schedule
            current_sora_sparse_lambda_2 = args.sora_sparse_lambda_2

            if planned_total_steps < 10000:
                if current_sora_sparse_lambda_2 >= 1e-3:
                    current_sora_sparse_lambda_2 = 1e-4

            for method in args.methods:
                seed_results: List[Dict[str, Any]] = []
                method_warmup_ratio = float(args.warmup_ratio)
                method_lr_scheduler_type = "linear"
                method_lr = float(args.lr)
                if method == "toplora":
                    # TopLoRA 官方脚本 lr=1e-4（Qwen2.5-3B+math_10k），GLUE 8e-4 下观测到崩溃。
                    # 原因：λ(x) = exp(RMSNorm(x @ W_λ)) 的 exp 非线性在高 lr 长训练下容易爆炸。
                    if getattr(args, "toplora_lr", None) is not None:
                        method_lr = float(args.toplora_lr)
                        if args.is_main_process:
                            print(f"[toplora] 使用 --toplora_lr={method_lr} 覆盖全局 lr={args.lr}")
                    elif float(args.lr) > 2e-4:
                        method_lr = max(float(args.lr) / 4.0, 1e-4)
                        if args.is_main_process:
                            print(
                                f"[toplora] 自动降低学习率: {args.lr} -> {method_lr} "
                                f"(TopLoRA 门控 exp(·) 在高 lr 长训练下易爆炸；"
                                f"官方脚本 lr=1e-4；可通过 --toplora_lr 显式覆盖)"
                            )
                if method == "pissa":
                    # PiSSA 官方脚本 lr=2e-5（Llama-2-7b+MetaMath），GLUE 小任务经验值 1e-4~2e-4。
                    # 原因：PiSSA 的 SVD 初始化使 ||A||,||B|| ~ 7（LoRA 仅 ~0.01 + 0），
                    # 用与 LoRA 相同 lr 会让 grad_norm 超过 clip 阈值数倍，主成分方向失真。
                    if getattr(args, "pissa_lr", None) is not None:
                        method_lr = float(args.pissa_lr)
                        if args.is_main_process:
                            print(f"[pissa] 使用 --pissa_lr={method_lr} 覆盖全局 lr={args.lr}")
                    elif float(args.lr) > 2e-4:
                        # 全局 lr 偏高时自动降为 lr/4，但下限 1e-4；显式 --pissa_lr 可覆盖该行为
                        method_lr = max(float(args.lr) / 4.0, 1e-4)
                        if args.is_main_process:
                            print(
                                f"[pissa] 自动降低学习率: {args.lr} -> {method_lr} "
                                f"(PiSSA 对 lr 敏感，官方 2e-5，本任务 lr>2e-4 时默认 lr/4；"
                                f"可通过 --pissa_lr 显式覆盖)"
                            )
                for seed in seeds:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)

                    # 每个 seed 重建 DataLoader，确保 DistributedSampler 使用当前 seed
                    # 而非固定的 seeds[0]，保证多 seed 实验采样随机性相互独立。
                    train_loader, val_loader, val_loader_eval_full = make_loaders(seed)

                    method_model = copy.deepcopy(base_model)
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
                        task_type=args.task_type,
                        is_main_process=args.is_main_process,
                        ddp_enabled=args.ddp_enabled,
                        nlu_regression=nlu_is_glue_regression(task_name),
                        lora_alpha=args.lora_alpha,
                        target_modules_override=args.target_modules,
                        module_preset=args.module_preset,
                        comparison_protocol=args.comparison_protocol,
                        protocol_dropout=args.protocol_dropout,
                        evorank_r_max=args.evorank_r_max,
                        evorank_use_rslora=args.evorank_use_rslora,
                        evorank_alpha_u=args.evo_alpha_u,
                        evorank_beta_u=args.evo_beta_u,
                        evorank_rho=args.evo_rho,
                        evorank_p_g=args.evo_p_g,
                        evorank_p_p=args.evo_p_p,
                        evorank_H_g=args.evo_H_g,
                        evorank_H_p=args.evo_H_p,
                        evorank_cooldown_steps=args.evo_cooldown_steps,
                        evorank_allow_reallocation=args.evo_allow_reallocation,
                        evorank_max_reallocate_candidates=args.evo_max_reallocate_candidates,
                        evorank_compensation_mode=args.evo_compensation_mode, # 传递评价中使用的补偿策略
                        toplora_dropout=args.toplora_dropout,
                        toplora_lambda_clamp=args.toplora_lambda_clamp,
                    )
                    # === PiSSA / Flat-LoRA 初始化诊断 ===
                    if args.is_main_process and method in ("pissa", "flatlora"):
                        _diag_inner = _unwrap_for_save(method_model)
                        _diag_lora_delta_norms: List[float] = []
                        _diag_base_norms: List[float] = []
                        _diag_scaling_val: Optional[float] = None
                        _diag_checked = 0
                        for _dn, _dm in _diag_inner.named_modules():
                            if not (hasattr(_dm, "lora_A") and hasattr(_dm, "lora_B")):
                                continue
                            try:
                                if isinstance(_dm.lora_A, (dict, nn.ModuleDict)):
                                    _ak = "default" if "default" in _dm.lora_A else next(iter(_dm.lora_A.keys()))
                                    _dA = _dm.lora_A[_ak].weight.data
                                    _dB = _dm.lora_B[_ak].weight.data
                                    _ds = float(_dm.scaling[_ak])
                                else:
                                    continue
                                if _diag_scaling_val is None:
                                    _diag_scaling_val = _ds
                                _delta = _ds * (_dB @ _dA)
                                _diag_lora_delta_norms.append(float(_delta.float().norm().item()))
                                if hasattr(_dm, "get_base_layer"):
                                    _diag_base_norms.append(float(_dm.get_base_layer().weight.data.float().norm().item()))
                                _diag_checked += 1
                            except Exception:
                                continue
                        if _diag_checked > 0:
                            import statistics as _st
                            _dn_mean = _st.mean(_diag_lora_delta_norms)
                            _dn_max = max(_diag_lora_delta_norms)
                            _bn_mean = _st.mean(_diag_base_norms) if _diag_base_norms else 0
                            print(
                                f"[{method}][init-diag] task={task_name} seed={seed} "
                                f"scaling(alpha/r)={_diag_scaling_val:.2f} "
                                f"||scaling*B@A||: mean={_dn_mean:.4f} max={_dn_max:.4f}  "
                                f"||W_base||: mean={_bn_mean:.4f}  layers={_diag_checked}"
                            )
                            if method == "pissa" and _diag_scaling_val is not None and abs(_diag_scaling_val - 1.0) > 0.01:
                                print(
                                    f"[pissa][WARNING] scaling={_diag_scaling_val:.2f} != 1.0 — "
                                    f"PiSSA 论文要求 lora_alpha == r 以保证初始 effective weight == pretrained weight。"
                                    f"当前 alpha={args.lora_alpha}, r={args.target_rank}。"
                                    f"若 PEFT 未在 SVD 分解中补偿 scaling，初始有效权重将偏离预训练 {abs(_diag_scaling_val - 1):.0%}×(top-r SVD)。"
                                )
                        if method == "pissa" and args.is_main_process:
                            _init_a_norms, _init_b_norms = [], []
                            for _n, _p in method_model.named_parameters():
                                if not _p.requires_grad:
                                    continue
                                if "lora_A" in _n:
                                    _init_a_norms.append(float(_p.detach().norm().item()))
                                elif "lora_B" in _n:
                                    _init_b_norms.append(float(_p.detach().norm().item()))
                            _ia_mean = sum(_init_a_norms) / max(len(_init_a_norms), 1)
                            _ib_mean = sum(_init_b_norms) / max(len(_init_b_norms), 1)
                            _ia_max = max(_init_a_norms) if _init_a_norms else 0.0
                            _ib_max = max(_init_b_norms) if _init_b_norms else 0.0
                            print(
                                f"[pissa][init-params] ||A||: mean={_ia_mean:.4f} max={_ia_max:.4f} count={len(_init_a_norms)}  "
                                f"||B||: mean={_ib_mean:.4f} max={_ib_max:.4f} count={len(_init_b_norms)}"
                            )
                            print(
                                f"[pissa][init-params] 论文推荐 CoLA: lr=1e-4, bs=16, epochs=20, alpha=8  "
                                f"当前: lr={method_lr}, bs={args.batch_size}, epochs={args.epochs}, alpha={args.lora_alpha}"
                            )

                    checkpoint_root: Optional[str] = None
                    if not args.no_output_dir:
                        safe_model_name = model_name.replace("/", "_").replace("\\", "_")
                        seed_suffix = f"_seed{seed}" if len(seeds) > 1 else ""
                        checkpoint_root = os.path.join(
                            args.output_dir, f"{task_name}_{safe_model_name}_{method}{seed_suffix}"
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
                        lr=method_lr,
                        head_lr=args.head_lr,
                        weight_decay=args.weight_decay,
                        warmup_ratio=method_warmup_ratio,
                        lr_scheduler_type=method_lr_scheduler_type,
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
                        random_seed=seed,
                        val_loader_eval_full=val_loader_eval_full,
                        sora_sparse_lambda=args.sora_sparse_lambda,
                        sora_sparse_lambda_2=current_sora_sparse_lambda_2,
                        sora_lambda_warmup_steps=args.sora_lambda_warmup_steps,
                        sora_lambda_schedule=current_sora_lambda_schedule,
                        sora_max_lambda=args.sora_max_lambda,
                        sora_lambda_num=args.sora_lambda_num,
                        task_name=task_name,
                        checkpoint_root=checkpoint_root,
                        save_steps=args.save_steps,
                        save_every_epoch=args.save_every_epoch,
                        save_final_model=not args.no_save_final_model,
                        verify_n_samples=args.verify_n_samples,
                        max_grad_norm=args.max_grad_norm,
                        flatlora_rho=args.flatlora_rho,
                        peft_meta=meta,
                        evo_include_noop_candidate=args.evo_include_noop_candidate,
                    )
                    res.update(
                        {
                            "task": task_name,
                            "backbone": model_name,
                            "target_rank": args.target_rank,
                            "trainable_params": meta["trainable_params"],
                            "extra_params": meta.get("extra_params", 0),
                            "target_modules": ",".join(meta.get("target_modules", [])),
                            "effective_dropout": (
                                meta.get("effective_dropout", "")
                                if meta.get("effective_dropout", None) is not None
                                else ""
                            ),
                            "seed": seed,
                        }
                    )
                    if args.is_main_process:
                        all_results.append(res)
                        seed_results.append(res)

                # 多种子聚合：追加均值/标准差汇总行
                if args.is_main_process and len(seed_results) > 1:
                    metric_keys = [
                        "matthews_corrcoef",
                        "accuracy",
                        "accuracy_m",
                        "accuracy_mm",
                        "accuracy_m_mm_mean",
                        "accuracy_f1_mean",
                        "f1",
                        "pearson_spearman_mean",
                        "pearson",
                        "spearman",
                        "rouge1",
                        "rouge2",
                        "rougeL",
                    ]
                    
                    mean_row = dict(seed_results[0])
                    std_row = dict(seed_results[0])
                    mean_row["seed"] = "mean"
                    std_row["seed"] = "std"

                    for m_key in metric_keys:
                        vals = [
                            float(r[m_key])
                            for r in seed_results
                            if m_key in r and isinstance(r[m_key], (int, float)) and not isinstance(r[m_key], bool)
                        ]
                        if vals:
                            mean_row[m_key] = statistics.mean(vals)
                            std_row[m_key] = statistics.stdev(vals) if len(vals) > 1 else 0.0
                        else:
                            mean_row[m_key] = ""
                            std_row[m_key] = ""
                            
                    # 清空不适合聚合的字段
                    for row in (mean_row, std_row):
                        row["artifact_dir"] = ""
                        row["final_dir"] = ""
                        row["peak_memory_mb"] = ""
                        row["total_train_time_sec"] = ""
                        row["avg_active_rank"] = ""
                        row["warmup_ratio"] = ""
                        row["lr_scheduler_type"] = ""
                        row["evorank_avg_delta_complexity"] = ""
                    all_results.extend([mean_row, std_row])

    if args.is_main_process:
        with open(args.export_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "task",
                    "backbone",
                    "method",
                    "seed",
                    "val_metric_key",
                    "target_rank",
                    "trainable_params",
                    "extra_params",
                    "target_modules",
                    "effective_dropout",
                    "optimizer_type",
                    "matthews_corrcoef",
                    "accuracy",
                    "accuracy_m",
                    "accuracy_mm",
                    "accuracy_m_mm_mean",
                    "accuracy_f1_mean",
                    "f1",
                    "pearson_spearman_mean",
                    "pearson",
                    "spearman",
                    "rouge1",
                    "rouge2",
                    "rougeL",
                    "peak_memory_mb",
                    "avg_active_rank",
                    "warmup_ratio",
                    "lr_scheduler_type",
                    "evorank_avg_delta_complexity",
                    "total_train_time_sec",
                    "artifact_dir",
                    "final_dir",
                ],
                extrasaction="ignore",
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
        if args.ddp_enabled and torch.cuda.is_available():
            # 在首次 collective 之前绑定本地 GPU，
            # 避免 NCCL 提示 "devices used by this process are currently unknown"。
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(
                backend=args.ddp_backend,
                timeout=datetime.timedelta(seconds=1800),
            )

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        results = run_protocol_grid(args)

        if args.is_main_process:
            if results:
                mk = (results[0].get("val_metric_key") or "").strip()
                if mk:
                    print(
                        f"[summary] 验证主指标日志键为 val_{mk}；该值已落入 CSV 对应列。"
                    )
            print("\n=== Benchmark Summary ===")
            print(
                f"{'task':<8} {'backbone':<16} {'seed':<6} {'method':<12} {'best_metric':<12} "
                f"{'peak_mem_mb':<12} {'avg_rank':<10} {'time_sec':<10}"
            )
            for r in results:
                ar = r["avg_active_rank"]
                ar_fmt = f"{float(ar):.4f}" if isinstance(ar, float) else str(ar)
                
                eval_key = r.get("val_metric_key", "")
                best_val = r.get(eval_key, 0.0) if eval_key else 0.0
                if best_val == "" or best_val == "N/A":
                    best_val = 0.0

                pm = r.get('peak_memory_mb', 0.0)
                pm_fmt = f"{float(pm):<12.2f}" if pm != "" else f"{pm:<12}"
                tt = r.get('total_train_time_sec', 0.0)
                tt_fmt = f"{float(tt):<10.2f}" if tt != "" else f"{tt:<10}"

                print(
                    f"{r.get('task', args.task_name):<8} "
                    f"{r.get('backbone', args.model_name):<16} "
                    f"{str(r.get('seed', '')):<6} "
                    f"{r['method']:<12} "
                    f"{float(best_val):<12.4f} "
                    f"{pm_fmt} "
                    f"{ar_fmt:<10} "
                    f"{tt_fmt}"
                )
    finally:
        if args.ddp_enabled and dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
