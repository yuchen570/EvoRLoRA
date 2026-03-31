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

from glue_metrics import collect_nlu_predictions, compute_glue_primary_metric, glue_primary_metric_key, compute_glue_metrics_dict

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

        val_loader_eval_full: Optional[DataLoader] = None
        if ddp_enabled and world_size > 1:
            train_sampler = DistributedSampler(
                tokenized[train_split],
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
                drop_last=False,
            )
            # 为了让 mini_val_k 的批次数在各卡上尽量一致，这里使用 drop_last=True。
            val_sampler = DistributedSampler(
                tokenized[val_split],
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=seed,
                drop_last=True,
            )
            train_loader = DataLoader(
                tokenized[train_split], batch_size=batch_size, sampler=train_sampler, collate_fn=collator
            )
            val_loader = DataLoader(
                tokenized[val_split], batch_size=batch_size, sampler=val_sampler, collate_fn=collator
            )
            # GLUE 官方主指标（MCC/F1 等）需全验证集；与 NLG 一致，仅 rank0 用该 loader 做评估。
            val_loader_eval_full = DataLoader(
                tokenized[val_split], batch_size=batch_size, shuffle=False, collate_fn=collator
            )
        else:
            train_loader = DataLoader(tokenized[train_split], batch_size=batch_size, shuffle=True, collate_fn=collator)
            val_loader = DataLoader(tokenized[val_split], batch_size=batch_size, shuffle=False, collate_fn=collator)

        if task_name == "stsb":
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1,
                problem_type="regression",
                cache_dir=model_cache_dir,
            )
        else:
            label_feature = dataset[train_split].features.get("label", None)
            if label_feature is not None and hasattr(label_feature, "num_classes") and label_feature.num_classes is not None:
                num_labels = int(label_feature.num_classes)
            else:
                num_labels = len(set(dataset[train_split]["label"]))
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                cache_dir=model_cache_dir,
            )
        return train_loader, val_loader, val_loader_eval_full, base_model, tokenizer

    if task_type == "nlg":
        if nlg_dataset_name == "cnn_dailymail":
            dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir=dataset_cache_dir)
            text_key = "article"
            target_key = "highlights"
        elif nlg_dataset_name == "xsum":
            dataset = load_dataset("xsum", cache_dir=dataset_cache_dir)
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
    lora_alpha: Optional[float] = None,
    target_modules_override: Optional[str] = None,
    lora_ga_use_rslora: bool = False,
    lora_ga_stable_gamma: Optional[float] = None,
) -> Tuple[nn.Module, Optional[RankEvolutionController], Dict[str, Any]]:
    # --target_modules 优先；否则按模型类型自动推断
    if target_modules_override:
        target_modules = [m.strip() for m in target_modules_override.split(",") if m.strip()]
    else:
        model_type = getattr(getattr(model, "config", None), "model_type", "").lower()
        if "deberta" in model_type:
            target_modules = ["query_proj", "key_proj", "value_proj"]
        elif "roberta" in model_type or "bert" in model_type:
            target_modules = ["query", "value"]
        elif "llama" in model_type or "mistral" in model_type:
            target_modules = ["q_proj", "v_proj"]
        elif "t5" in model_type:
            target_modules = ["q", "v"]
        else:
            target_modules = ["query", "value"]

    # --lora_alpha 优先；否则回退到 2 * target_rank
    effective_alpha = float(lora_alpha) if lora_alpha is not None else float(2 * target_rank)

    model_type = getattr(getattr(model, "config", None), "model_type", "").lower()

    # PEFT 的 task_type 需要与 backbone 类型匹配，避免 seq2seq/causal 分支内部逻辑错误。
    if "t5" in model_type:
        peft_task_type = TaskType.SEQ_2_SEQ_LM
    elif "llama" in model_type or "mistral" in model_type:
        peft_task_type = TaskType.CAUSAL_LM
    else:
        peft_task_type = TaskType.SEQ_CLS

    controller: Optional[RankEvolutionController] = None

    if task_type == "nlu":
        modules_to_save = ["classifier", "score", "pooler"]
    else:
        modules_to_save = None

    if method_name == "lora":
        config = LoraConfig(
            task_type=peft_task_type,
            r=target_rank,
            lora_alpha=effective_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
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
            tfinal = max(int(0.1 * planned_steps), tinit + 1)
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
            lora_alpha=effective_alpha,
            lora_dropout=0.1,
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
            if planned_steps < 10000:
                # 对极小数据集移除正交惩罚，采用极小值 1e-8 以满足 PEFT 底层不能 <=0 的硬性断言要求
                tgt_orth = 1e-8
            
            if tgt_orth <= 0:
                print("Warning: PEFT AdaLoRA requires orth_reg_weight > 0. Clamping to 1e-8 to avoid crash.")
                tgt_orth = 1e-8
                
            adalora_kw["orth_reg_weight"] = tgt_orth
        config = AdaLoraConfig(**adalora_kw)
        model = get_peft_model(model, config)

    elif method_name == "evorank":
        controller = inject_evo_lora(
            model=model,
            target_modules=target_modules,
            layer_kwargs={"r_max": 16, "r_init": target_rank, "lora_alpha": effective_alpha},
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
            stable_gamma=lora_ga_stable_gamma,
        )
        config = LoraConfig(
            task_type=peft_task_type,
            r=target_rank,
            lora_alpha=effective_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            bias="none",
            use_rslora=lora_ga_use_rslora,
        )
        model = get_peft_model(model, config)
        tgt = next(model.parameters()).device
        apply_lora_ga_init_to_peft(model, init_by_key, target_device=tgt)

    elif method_name == "sora":
        inject_sora(
            model=model,
            target_modules=target_modules,
            r=target_rank,
            lora_alpha=effective_alpha,
            lora_dropout=0.1,
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
        # LoRA / LoRA-GA: 固定秩
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
    
    if method_name not in ("lora", "lora-ga"):
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
    head_lr: Optional[float] = None,
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
    head_lr_val = head_lr if head_lr is not None else max(lr, 5e-4)

    if method_name == "sora":
        # 官方 SoRA：gate 参数使用独立的 SparseAdamW（近端梯度），其余参数使用标准 AdamW。
        _non_gate_peft = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith(".gate") and not any(k in n for k in ["classifier", "score", "lm_head", "shared", "pooler"])]
        _non_gate_head = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith(".gate") and any(k in n for k in ["classifier", "score", "lm_head", "shared", "pooler"])]
        _gate = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith(".gate")]
        
        # 兼容 SoRA 的官方实现限制 (动态修正为 0.1 阻值)
        sora_wd = 0.1 if weight_decay == 0.01 else weight_decay

        optimizer = AdamW([
            {"params": _non_gate_peft, "lr": lr},
            {"params": _non_gate_head, "lr": head_lr_val}
        ], weight_decay=sora_wd)
        sparse_optimizer = SparseAdamW(_gate, lr=lr, sparse_lambda=sora_sparse_lambda_2, weight_decay=0.0)
    else:
        _peft_params = [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in ["classifier", "score", "lm_head", "shared", "pooler"])]
        _head_params = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in ["classifier", "score", "lm_head", "shared", "pooler"])]
        
        # LoRA-GA 完全依靠 A、B 矩阵精准抵消基础权重中巨大的负向偏移。
        # 权重衰减会压缩 A、B 矩阵，瞬间破坏这种脆弱的平衡并损坏模型。
        # 因此，LoRA-GA 参数的 weight_decay 必须设为 0.0，分类头可以保留权重衰减。
        dynamic_wd_peft = 0.0 if method_name == "lora-ga" else weight_decay

        optimizer = AdamW([
            {"params": _peft_params, "lr": lr, "weight_decay": dynamic_wd_peft},
            {"params": _head_params, "lr": head_lr_val}
        ], weight_decay=weight_decay)
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
    best_val_metrics: Dict[str, float] = {}
    train_loss_ema = None
    ema_beta = 0.95
    rouge1_val: float = 0.0
    rouge2_val: float = 0.0

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
                    max_grad_norm=max_grad_norm,
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
                    if sora_lambda_schedule is not None:
                        # schedule-dense 变体：按 schedule 动态调整 sparse_lambda
                        if sora_lambda_schedule == "linear":
                            num_steps = max(int(sora_lambda_num), 1)
                            lam = float(sora_max_lambda) * min(1.0, float(global_step + 1) / float(num_steps))
                    elif sora_lambda_warmup_steps > 0:
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
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_grad_norm
                    )
                if method_name == "sora":
                    # 防止 gate 梯度中的 NaN/Inf 传入优化器状态，导致后续参数全 NaN。
                    for n, p in model.named_parameters():
                        if n.endswith(".gate") and p.grad is not None and not torch.isfinite(p.grad).all():
                            p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                optimizer.step()
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
            metrics_dict_val = {}
            if is_main_process:
                ev_loader = val_loader_eval_full if val_loader_eval_full is not None else val_loader
                eval_model = _unwrap_training_module(model)
                y_pred, y_true = collect_nlu_predictions(eval_model, ev_loader, device, regression=regression)
                val_metric = compute_glue_primary_metric(task_name, y_pred, y_true)
                metrics_dict_val = compute_glue_metrics_dict(task_name, y_pred, y_true)
            if ddp_enabled and dist.is_available() and dist.is_initialized():
                m_tensor = torch.tensor([val_metric], device=device, dtype=torch.float64)
                dist.broadcast(m_tensor, src=0)
                val_metric = float(m_tensor.item())
                # DDP环境下，这里只广播主指标。字典仅由is_main_process维护！
                dist.barrier()
            if val_metric > best_val_acc:
                best_val_acc = val_metric
                best_val_metrics = metrics_dict_val
            elif val_metric == best_val_acc and not best_val_metrics:
                best_val_metrics = metrics_dict_val
            if is_main_process:
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
                    )) if method_name in ("lora", "lora-ga") else 8,
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
            # DDP 下用 barrier 对齐：rank0 在完整验证集上算 ROUGE，避免 DistributedSampler 子集有偏。
            if ddp_enabled and dist.is_available() and dist.is_initialized():
                dist.barrier()

            val_metric = 0.0
            if is_main_process:
                try:
                    import evaluate  # type: ignore

                    rouge_metric = evaluate.load("rouge")
                except Exception as e:
                    # NLG 指标在一些离线/依赖缺失环境下可能无法加载 rouge。
                    # 之前静默写 0 分，导致“全 0”难以定位原因；这里显式打印异常。
                    rouge_metric = None
                    print(f"[warn] evaluate.load('rouge') failed: {type(e).__name__}: {e!r}")

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
                    )) if method_name in ("lora", "lora-ga") else 8,
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

            if ddp_enabled and dist.is_available() and dist.is_initialized():
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

    result = {
        "method": method_name,
        "total_train_time_sec": total_time,
        "peak_memory_mb": peak_mem_mb,
        "avg_active_rank": final_avg_active_rank,
        "artifact_dir": artifact_dir_str,
        "final_dir": final_dir_str,
        "val_metric_key": val_metric_key_str,
        
        "matthews_corrcoef": best_val_metrics.get("matthews_corrcoef", "") if task_type == "nlu" else "",
        "accuracy": best_val_metrics.get("accuracy", "") if task_type == "nlu" else "",
        "f1": best_val_metrics.get("f1", "") if task_type == "nlu" else "",
        "pearson_spearman_mean": best_val_metrics.get("pearson_spearman_mean", "") if task_type == "nlu" else "",
        "pearson": best_val_metrics.get("pearson", "") if task_type == "nlu" else "",
        "spearman": best_val_metrics.get("spearman", "") if task_type == "nlu" else "",
        "rouge1": rouge1_val if task_type == "nlg" else "",
        "rouge2": rouge2_val if task_type == "nlg" else "",
        "rougeL": best_val_acc if val_metric_key_str == "rougeL" else "",
    }
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
    parser.add_argument("--methods", nargs="+", default=["lora", "adalora", "evorank"])
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
             "未指定时按模型类型自动推断（DeBERTa→query_proj/key_proj/value_proj，RoBERTa/BERT→query/value，T5→q/v）。",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=None, help="分类头或额外可训练参数的学习率。None时默认 max(lr, 5e-4) 以解决 CE 任务不收敛问题")
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
    parser.add_argument("--lora_ga_batches", type=int, default=8, help="LoRA-GA 梯度估计用的训练 batch 数（仅前若干个 batch）")
    parser.add_argument("--lora_ga_use_rslora", action="store_true", help="LoRA-GA：启用 rsLoRA 缩放（lora_alpha/sqrt(r)），对标官方 reproduce 配置")
    parser.add_argument("--lora_ga_stable_gamma", type=float, default=None, help="LoRA-GA：stable init gamma 值（官方默认 64）；指定后在 SVD 初始化时对奇异值做 gamma 归一化")
    parser.add_argument("--sora_sparse_lambda", type=float, default=1e-3, help="SoRA：gate L1 惩罚系数")
    parser.add_argument("--sora_lambda_warmup_steps", type=int, default=0, help="SoRA：前若干步将 sparse_lambda 从 0 线性升到设定值；0 表示关闭")
    parser.add_argument("--sora_sparse_lambda_2", type=float, default=3e-4, help="SoRA：gate 近端梯度硬裁剪阈值（Proximal Gradient / Soft-Thresholding），对标官方 SparseAdamW")
    parser.add_argument(
        "--sora_lambda_schedule",
        type=str,
        default=None,
        help="SoRA lambda 调度策略（如 'linear'）。None 表示固定值（no-schedule 主线）。",
    )
    parser.add_argument("--sora_max_lambda", type=float, default=10.0, help="SoRA lambda_schedule 的最大 lambda 值")
    parser.add_argument("--sora_lambda_num", type=int, default=5, help="SoRA lambda_schedule 的步数（线性升至 max_lambda 所需步数）")
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
    return parser.parse_args()


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
            # 数据加载在种子循环外（数据集与种子无关，避免重复下载/tokenize）
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
                seed=seeds[0],
            )
            planned_total_steps = (
                args.max_train_steps if args.max_train_steps is not None else args.epochs * len(train_loader)
            )

            current_sora_lambda_schedule = args.sora_lambda_schedule
            current_sora_sparse_lambda_2 = args.sora_sparse_lambda_2
            current_lora_ga_batches = args.lora_ga_batches
            current_lora_ga_stable_gamma = args.lora_ga_stable_gamma

            if planned_total_steps < 10000:
                if current_sora_sparse_lambda_2 >= 1e-3:
                    current_sora_sparse_lambda_2 = 1e-4
                if current_lora_ga_batches < 32:
                    current_lora_ga_batches = min(32, len(train_loader) if len(train_loader) > 0 else 32)
                if current_lora_ga_stable_gamma is None:
                    pass

            for method in args.methods:
                seed_results: List[Dict[str, Any]] = []
                for seed in seeds:
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)

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
                        lora_ga_batches=current_lora_ga_batches,
                        task_type=args.task_type,
                        lora_ga_device=lora_ga_dev,
                        is_main_process=args.is_main_process,
                        ddp_enabled=args.ddp_enabled,
                        nlu_regression=nlu_is_glue_regression(task_name),
                        lora_alpha=args.lora_alpha,
                        target_modules_override=args.target_modules,
                        lora_ga_use_rslora=args.lora_ga_use_rslora,
                        lora_ga_stable_gamma=current_lora_ga_stable_gamma,
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
                        lr=args.lr,
                        head_lr=args.head_lr,
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
                    )
                    res.update(
                        {
                            "task": task_name,
                            "backbone": model_name,
                            "trainable_params": meta["trainable_params"],
                            "seed": seed,
                        }
                    )
                    if args.is_main_process:
                        all_results.append(res)
                        seed_results.append(res)

                # 多种子聚合：追加均值/标准差汇总行
                if args.is_main_process and len(seed_results) > 1:
                    metric_keys = ["matthews_corrcoef", "accuracy", "f1", "pearson_spearman_mean", "pearson", "spearman", "rouge1", "rouge2", "rougeL"]
                    
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
                    "trainable_params",
                    "matthews_corrcoef",
                    "accuracy",
                    "f1",
                    "pearson_spearman_mean",
                    "pearson",
                    "spearman",
                    "rouge1",
                    "rouge2",
                    "rougeL",
                    "peak_memory_mb",
                    "avg_active_rank",
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
                        f"[summary] 验证主指标日志键为 val_{mk}；该值已落入 CSV 对应列。"
                    )
            print("\n=== Benchmark Summary ===")
            print(
                f"{'task':<8} {'backbone':<16} {'method':<12} {'best_metric':<12} "
                f"{'peak_mem_mb':<12} {'avg_rank':<10} {'time_sec':<10}"
            )
            for r in results:
                ar = r["avg_active_rank"]
                ar_fmt = f"{float(ar):.4f}" if isinstance(ar, float) else str(ar)
                
                eval_key = r.get("val_metric_key", "")
                best_val = r.get(eval_key, 0.0) if eval_key else 0.0
                if best_val == "" or best_val == "N/A":
                    best_val = 0.0

                print(
                    f"{r.get('task', args.task_name):<8} "
                    f"{r.get('backbone', args.model_name):<16} "
                    f"{r['method']:<12} "
                    f"{float(best_val):<12.4f} "
                    f"{r.get('peak_memory_mb', 0.0):<12.2f} "
                    f"{ar_fmt:<10} "
                    f"{r.get('total_train_time_sec', 0.0):<10.2f}"
                )
    finally:
        if args.ddp_enabled and dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
