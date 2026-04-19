"""
SQuAD v1.1 / v2.0 抽取式问答基准（对齐 AdaLoRA 论文 Table 2）。

- 支持方法: lora / lora_kaiming / pissa / adalora / evorank / sora / flatlora / toplora
- 指标: Exact Match (EM) / F1（HF ``evaluate``: ``squad`` 或 ``squad_v2``）
- 写入:
    * ``<output_dir>/eval_results.json`` — 单次运行的指标与元数据
    * ``--export_csv`` 追加一行（方便多次运行聚合）

主要参考
--------
- HF 官方 ``examples/pytorch/question-answering/run_qa.py``
- AdaLoRA 论文 Table 2: DeBERTaV3-base × SQuAD v1.1 / v2.0 × 4 参数预算
    (0.08% / 0.16% / 0.32% / 0.65%, 分别对应 rank r ∈ {1, 2, 4, 8})

典型用法
--------
    torchrun --nproc_per_node=2 --master_port=29550 run_qa_benchmark.py \
        --model_name_or_path microsoft/deberta-v3-base \
        --method lora --dataset_name squad \
        --max_seq_length 384 --doc_stride 128 \
        --per_device_train_batch_size 16 --learning_rate 1e-3 \
        --num_train_epochs 3 --lora_rank 8 --lora_alpha 16 \
        --output_dir artifacts/qa/squad_deberta_lora_r8_s42 \
        --export_csv results_fair_qa_squadv1.csv --seed 42
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import math
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import load_dataset
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import transformers
from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model

from train_integration import inject_evo_lora
from adalora_utils import (
    normalize_adalora_schedule,
    unwrap_inner_from_training_model,
)
from sora_inject import inject_sora, SparseAdamW
from flatlora_inject import FlatLoRAHookManager
from toplora_inject import inject_toplora
from hf_cache_utils import check_local_dataset, resolve_pretrained_model_source

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.makedirs("./models", exist_ok=True)
os.makedirs("./datasets", exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.abspath("./models"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.abspath("./models"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.abspath("./datasets"))


# ---------------------------------------------------------------------------
# 1) 目标模块默认值（DeBERTa 系列已与 run_benchmark.py 对齐）
# ---------------------------------------------------------------------------

def _default_target_modules(model_type: str) -> List[str]:
    m = model_type.lower()
    if "deberta" in m:
        return ["query_proj", "key_proj", "value_proj",
                "attention.output.dense", "intermediate.dense", "output.dense"]
    if "bert" in m or "roberta" in m:
        return ["query", "key", "value"]
    return ["query_proj", "key_proj", "value_proj"]


# ---------------------------------------------------------------------------
# 2) 数据预处理：滑窗 tokenization（训练 / 验证两套 map 函数）
# ---------------------------------------------------------------------------

def build_qa_preprocessors(
    tokenizer: transformers.PreTrainedTokenizerBase,
    *,
    max_seq_length: int,
    doc_stride: int,
    pad_to_max_length: bool,
    version_2_with_negative: bool,
    question_col: str = "question",
    context_col: str = "context",
    answer_col: str = "answers",
) -> Tuple[Callable[[Dict[str, Any]], Dict[str, Any]],
           Callable[[Dict[str, Any]], Dict[str, Any]]]:
    pad_on_right = tokenizer.padding_side == "right"

    def prepare_train_features(examples: Dict[str, Any]) -> Dict[str, Any]:
        examples[question_col] = [q.lstrip() for q in examples[question_col]]
        tokenized = tokenizer(
            examples[question_col if pad_on_right else context_col],
            examples[context_col if pad_on_right else question_col],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
        )
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")
        tokenized["start_positions"] = []
        tokenized["end_positions"] = []
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples[answer_col][sample_index]
            if len(answers["answer_start"]) == 0:
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)
                continue
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            token_start_index = 0
            context_seq_id = 1 if pad_on_right else 0
            while sequence_ids[token_start_index] != context_seq_id:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != context_seq_id:
                token_end_index -= 1
            if not (offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char):
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)
            else:
                while (token_start_index < len(offsets)
                       and offsets[token_start_index][0] <= start_char):
                    token_start_index += 1
                tokenized["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized["end_positions"].append(token_end_index + 1)
        return tokenized

    def prepare_validation_features(examples: Dict[str, Any]) -> Dict[str, Any]:
        examples[question_col] = [q.lstrip() for q in examples[question_col]]
        tokenized = tokenizer(
            examples[question_col if pad_on_right else context_col],
            examples[context_col if pad_on_right else question_col],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
        )
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        tokenized["example_id"] = []
        context_idx = 1 if pad_on_right else 0
        for i in range(len(tokenized["input_ids"])):
            seq_ids = tokenized.sequence_ids(i)
            sample_index = sample_mapping[i]
            tokenized["example_id"].append(examples["id"][sample_index])
            tokenized["offset_mapping"][i] = [
                (o if seq_ids[k] == context_idx else None)
                for k, o in enumerate(tokenized["offset_mapping"][i])
            ]
        return tokenized

    _ = version_2_with_negative  # reserved; 阈值仅在 postprocess 阶段使用
    return prepare_train_features, prepare_validation_features


# ---------------------------------------------------------------------------
# 3) 后处理：start/end logits → 答案文本（完全对齐 HF utils_qa.postprocess_qa_predictions）
# ---------------------------------------------------------------------------

def postprocess_qa_predictions(
    examples,
    features,
    all_start_logits: np.ndarray,
    all_end_logits: np.ndarray,
    *,
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    if len(all_start_logits) != len(features):
        raise ValueError(f"logits={len(all_start_logits)} vs features={len(features)}")

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feat in enumerate(features):
        features_per_example[example_id_to_index[feat["example_id"]]].append(i)

    all_predictions: Dict[str, str] = collections.OrderedDict()
    scores_diff_json: Dict[str, float] = collections.OrderedDict()

    for example_index, example in enumerate(examples):
        feat_idxs = features_per_example[example_index]
        min_null_pred = None
        prelim: List[Dict[str, Any]] = []
        for fi in feat_idxs:
            s_logits = all_start_logits[fi]
            e_logits = all_end_logits[fi]
            offset_mapping = features[fi]["offset_mapping"]
            null_score = float(s_logits[0] + e_logits[0])
            if min_null_pred is None or min_null_pred["score"] > null_score:
                min_null_pred = {
                    "offsets": (0, 0), "score": null_score,
                    "start_logit": float(s_logits[0]), "end_logit": float(e_logits[0]),
                }
            s_idx = np.argsort(s_logits)[-1 : -n_best_size - 1 : -1].tolist()
            e_idx = np.argsort(e_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for si in s_idx:
                for ei in e_idx:
                    if (si >= len(offset_mapping) or ei >= len(offset_mapping)
                            or offset_mapping[si] is None
                            or offset_mapping[ei] is None
                            or len(offset_mapping[si]) < 2
                            or len(offset_mapping[ei]) < 2):
                        continue
                    if ei < si or ei - si + 1 > max_answer_length:
                        continue
                    prelim.append({
                        "offsets": (offset_mapping[si][0], offset_mapping[ei][1]),
                        "score": float(s_logits[si] + e_logits[ei]),
                        "start_logit": float(s_logits[si]),
                        "end_logit": float(e_logits[ei]),
                    })
        if version_2_with_negative and min_null_pred is not None:
            prelim.append(min_null_pred)
        preds = sorted(prelim, key=lambda x: x["score"], reverse=True)[:n_best_size]
        if (version_2_with_negative and min_null_pred is not None
                and not any(p["offsets"] == (0, 0) for p in preds)):
            preds.append(min_null_pred)
        ctx = example["context"]
        for p in preds:
            off = p.pop("offsets")
            p["text"] = ctx[off[0]:off[1]]
        if not preds or (len(preds) == 1 and preds[0]["text"] == ""):
            preds.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})
        if not version_2_with_negative:
            all_predictions[example["id"]] = preds[0]["text"]
        else:
            i = 0
            while i < len(preds) and preds[i]["text"] == "":
                i += 1
            best_non_null = preds[min(i, len(preds) - 1)]
            null_score = min_null_pred["score"] if min_null_pred else 0.0
            score_diff = null_score - best_non_null["start_logit"] - best_non_null["end_logit"]
            scores_diff_json[example["id"]] = float(score_diff)
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null["text"]
    return all_predictions, scores_diff_json


# ---------------------------------------------------------------------------
# 4) 模型注入：与 run_nlg_benchmark.py 保持一致，TaskType -> QUESTION_ANS
# ---------------------------------------------------------------------------

def build_qa_model_and_peft(args, method: str):
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model_load_id, use_local = resolve_pretrained_model_source(args.model_name_or_path, "./models")

    _prev_hub_offline = os.environ.get("HF_HUB_OFFLINE")
    _prev_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    if use_local:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if dist.is_initialized() and dist.get_rank() != 0:
        dist.barrier()

    try:
        model = transformers.AutoModelForQuestionAnswering.from_pretrained(
            model_load_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            cache_dir="./models",
            local_files_only=use_local,
        )
    finally:
        if _prev_hub_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = _prev_hub_offline
        if _prev_tf_offline is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = _prev_tf_offline

    if dist.is_initialized() and dist.get_rank() == 0:
        dist.barrier()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model_type = getattr(model.config, "model_type", "").lower() or args.model_name_or_path.lower()
    tms = args.target_modules.strip()
    target_modules = [m for m in tms.split(",") if m] if tms else _default_target_modules(model_type)
    controller = None

    if method in ("lora", "lora_kaiming", "pissa", "adalora", "flatlora"):
        if method == "pissa":
            init_weights = getattr(args, "pissa_init_method", "pissa_niter_16")
            dropout = 0.0
        else:
            init_weights = True  # True => kaiming_uniform A + zero B（含 lora_kaiming / flatlora）
            dropout = args.lora_dropout

        if method == "adalora":
            tinit_n, tfinal_n, _ = normalize_adalora_schedule(
                total_steps=max(1, getattr(args, "_total_steps", 1000)),
                adalora_tinit=args.adalora_tinit,
                adalora_tfinal=args.adalora_tfinal,
            )
            peft_config = AdaLoraConfig(
                task_type=TaskType.QUESTION_ANS,
                target_modules=target_modules,
                inference_mode=False,
                r=max(2, args.lora_rank * 2),
                target_r=args.lora_rank,
                init_r=max(2, args.lora_rank * 2),
                lora_alpha=args.lora_alpha,
                lora_dropout=dropout,
                tinit=int(tinit_n),
                tfinal=int(tfinal_n),
                deltaT=args.adalora_delta_t,
                beta1=0.85,
                beta2=0.85,
                orth_reg_weight=args.adalora_orth_reg_weight,
            )
        else:
            peft_config = LoraConfig(
                task_type=TaskType.QUESTION_ANS,
                target_modules=target_modules,
                inference_mode=False,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=dropout,
                init_lora_weights=init_weights,
            )
        logger.info(f"Initialize {method} adapter (QA) with r={args.lora_rank}, alpha={args.lora_alpha}, "
                    f"modules={target_modules}")
        model = get_peft_model(model, peft_config)

    elif method == "evorank":
        controller = inject_evo_lora(
            model=model,
            target_modules=target_modules,
            layer_kwargs={
                "r_max": args.lora_rank,
                "r_init": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
            },
            controller_kwargs={
                "rho": args.evorank_rho,
                "p_g": 1.0, "p_p": 0.05, "H_g": 4, "H_p": 4,
                "cooldown_steps": 100,
                "r_min": max(1, args.lora_rank // 4),
                "r_max": args.lora_rank,
                "alpha_u": 1.0, "beta_u": 0.5,
                "allow_reallocation": True,
                "expand_init_mode": "gradient",
                "max_reallocate_candidates": 16,
            },
        )
    elif method == "sora":
        inject_sora(model=model, target_modules=target_modules,
                    r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    elif method == "toplora":
        inject_toplora(model=model, target_modules=target_modules,
                       r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # dtype 策略：整体跟随 backbone dtype，避免 Linear/LayerNorm 混合 dtype 报错。
    # 只有在 float32 训练时，才把 QA 头 / 分类头 / norm 显式拉回 float32（通常 from_pretrained 已经是）。
    is_deberta = "deberta" in model_type
    if dtype == torch.float32:
        for name, module in model.named_modules():
            if "qa_outputs" in name or "classifier" in name:
                module.to(torch.float32)
            elif (not is_deberta) and "norm" in name.lower():
                module.to(torch.float32)
    for name, module in model.named_modules():
        if "lora_A" in name or "lora_B" in name or "lora_E" in name:
            module.to(dtype)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    meta = {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_pct": (trainable / total) if total > 0 else 0.0,
        "target_modules": target_modules,
    }
    return model, controller, meta


# ---------------------------------------------------------------------------
# 5) 评估：聚合 start/end logits → postprocess → squad / squad_v2 指标
# ---------------------------------------------------------------------------

def _pad_and_concat(chunks: List[np.ndarray], pad_value: float = -1e4) -> np.ndarray:
    """将变长的 [B_i, L_i] 数组在第二维右侧 pad 到相同长度后纵向拼接。"""
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    max_len = max(c.shape[1] for c in chunks)
    padded = []
    for c in chunks:
        if c.shape[1] < max_len:
            pad = np.full((c.shape[0], max_len - c.shape[1]), pad_value, dtype=c.dtype)
            c = np.concatenate([c, pad], axis=1)
        padded.append(c)
    return np.concatenate(padded, axis=0)


def evaluate_qa(
    model: nn.Module,
    eval_dataset,
    eval_examples,
    data_collator,
    device: torch.device,
    *,
    batch_size: int,
    version_2_with_negative: bool,
) -> Dict[str, Any]:
    model.eval()
    # 训练管线用的 dataset 已经 drop 了 example_id / offset_mapping 以外的原列；为送入模型，
    # 临时 strip 掉这两列再组 batch。
    columns_for_model = [c for c in eval_dataset.column_names
                         if c not in ("example_id", "offset_mapping")]
    model_input_ds = eval_dataset.with_format("torch", columns=columns_for_model)

    loader = DataLoader(
        model_input_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=False,
    )

    all_start: List[np.ndarray] = []
    all_end: List[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # AdaLoRA 的 .base_model 是个 ModulesToSaveWrapper / PeftModel wrapper；直接前向即可
            outputs = model(**batch)
            all_start.append(outputs.start_logits.detach().float().cpu().numpy())
            all_end.append(outputs.end_logits.detach().float().cpu().numpy())
    start_logits = _pad_and_concat(all_start)
    end_logits = _pad_and_concat(all_end)

    # 与 features 对齐所需的列
    features_cpu = eval_dataset.remove_columns(
        [c for c in eval_dataset.column_names if c not in ("example_id", "offset_mapping")]
    )
    preds, _ = postprocess_qa_predictions(
        examples=eval_examples,
        features=features_cpu,
        all_start_logits=start_logits,
        all_end_logits=end_logits,
        version_2_with_negative=version_2_with_negative,
    )

    # 走 HF evaluate
    try:
        import evaluate
        metric = evaluate.load("squad_v2" if version_2_with_negative else "squad")
    except Exception as e:
        logger.warning(f"evaluate 库不可用（{e!r}），使用内置简易 EM/F1 回退")
        metric = None

    if version_2_with_negative:
        formatted_preds = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
            for k, v in preds.items()
        ]
    else:
        formatted_preds = [{"id": k, "prediction_text": v} for k, v in preds.items()]
    refs = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]

    if metric is not None:
        res = metric.compute(predictions=formatted_preds, references=refs)
    else:
        res = _fallback_squad_metric(formatted_preds, refs)

    return {"metrics": res, "predictions": preds}


def _fallback_squad_metric(preds: List[Dict[str, Any]], refs: List[Dict[str, Any]]) -> Dict[str, float]:
    """零依赖备用实现：归一化+exact/F1，近似但足够判断收敛。"""
    import re, string
    def _norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = "".join(ch for ch in s if ch not in set(string.punctuation))
        s = re.sub(r"\s+", " ", s).strip()
        return s
    def _f1(p: str, g: str) -> float:
        pt, gt = _norm(p).split(), _norm(g).split()
        if not pt or not gt:
            return float(pt == gt)
        common = collections.Counter(pt) & collections.Counter(gt)
        nm = sum(common.values())
        if nm == 0:
            return 0.0
        precision = nm / len(pt); recall = nm / len(gt)
        return 2 * precision * recall / (precision + recall)
    em, f1 = 0.0, 0.0
    n = len(preds)
    ref_by_id = {r["id"]: r["answers"]["text"] for r in refs}
    for p in preds:
        golds = ref_by_id.get(p["id"], [""]) or [""]
        em += max(float(_norm(p["prediction_text"]) == _norm(g)) for g in golds)
        f1 += max(_f1(p["prediction_text"], g) for g in golds)
    return {"exact_match": 100.0 * em / max(1, n), "f1": 100.0 * f1 / max(1, n)}


# ---------------------------------------------------------------------------
# 6) 训练入口
# ---------------------------------------------------------------------------

def _cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(step: int):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def _load_squad_dataset(args) -> Any:
    name = args.dataset_name
    cfg = args.dataset_config_name
    offline = check_local_dataset(name, cfg, cache_dir="./datasets")
    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        ds = load_dataset(name, cfg, cache_dir="./datasets")
    finally:
        os.environ.pop("HF_DATASETS_OFFLINE", None)
    return ds


def run_qa_training(args, method: str):
    is_main = (not dist.is_initialized()) or (dist.get_rank() == 0)
    device = torch.device(f"cuda:{args.local_rank}" if args.local_rank != -1
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 先加载 tokenizer（QA 必须 fast tokenizer）
    model_load_id, use_local = resolve_pretrained_model_source(args.model_name_or_path, "./models")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_load_id,
        use_fast=True,
        cache_dir="./models",
        local_files_only=use_local,
        trust_remote_code=True,
    )
    if not isinstance(tokenizer, transformers.PreTrainedTokenizerFast):
        raise RuntimeError("抽取式 QA 需要 fast tokenizer（return_overflowing_tokens 依赖）")

    raw = _load_squad_dataset(args)
    if "train" not in raw or "validation" not in raw:
        raise RuntimeError(f"Dataset {args.dataset_name} 缺少 train / validation split")

    train_ex = raw["train"]
    val_ex = raw["validation"]
    if args.max_train_samples and args.max_train_samples > 0:
        train_ex = train_ex.select(range(min(args.max_train_samples, len(train_ex))))
    if args.max_eval_samples and args.max_eval_samples > 0:
        val_ex = val_ex.select(range(min(args.max_eval_samples, len(val_ex))))

    prep_train, prep_val = build_qa_preprocessors(
        tokenizer,
        max_seq_length=min(args.max_seq_length, tokenizer.model_max_length),
        doc_stride=args.doc_stride,
        pad_to_max_length=True,  # QA 走等长 pad，便于 collator
        version_2_with_negative=args.version_2_with_negative,
    )
    train_cols = train_ex.column_names
    val_cols = val_ex.column_names
    train_ds = train_ex.map(prep_train, batched=True, remove_columns=train_cols, desc="tokenizing train")
    eval_ds = val_ex.map(prep_val, batched=True, remove_columns=val_cols, desc="tokenizing eval")

    # 预估 total_steps，供 AdaLoRA 调度对齐使用
    sampler = DistributedSampler(train_ds, seed=args.seed) if dist.is_initialized() else None
    steps_per_epoch = math.ceil(len(train_ds) / max(1, args.per_device_train_batch_size *
                                                    (dist.get_world_size() if dist.is_initialized() else 1) *
                                                    args.gradient_accumulation_steps))
    total_steps = args.max_train_steps if args.max_train_steps and args.max_train_steps > 0 \
        else steps_per_epoch * args.num_train_epochs
    args._total_steps = total_steps

    model, controller, meta = build_qa_model_and_peft(args, method)
    model.to(device)

    # collator：QA 走简单 default_collator，所有列已 pad 等长
    data_collator = transformers.default_data_collator

    train_loader = DataLoader(
        train_ds.remove_columns([c for c in train_ds.column_names
                                 if c not in ("input_ids", "attention_mask",
                                              "token_type_ids", "start_positions", "end_positions")]),
        batch_size=args.per_device_train_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=data_collator,
        pin_memory=True,
    )

    warmup_steps = int(total_steps * args.warmup_ratio)
    opt_class = SparseAdamW if method == "sora" else torch.optim.AdamW
    optimizer = opt_class(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.learning_rate, weight_decay=args.weight_decay)
    sparse_optimizer = None
    if method == "sora":
        gate_params = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith(".gate")]
        normal_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith(".gate")]
        optimizer = torch.optim.AdamW(normal_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        sparse_optimizer = SparseAdamW(gate_params, lr=args.learning_rate,
                                       weight_decay=0.0, eps=1e-8)
    lr_sched = _cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    sparse_sched = _cosine_schedule_with_warmup(sparse_optimizer, warmup_steps, total_steps) \
        if sparse_optimizer is not None else None

    if dist.is_initialized():
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=False,
        )

    flatlora_mgr = FlatLoRAHookManager(model, args.flatlora_rho, total_steps) if method == "flatlora" else None

    if is_main:
        logger.info(f"*** QA training: method={method}, total_steps={total_steps}, warmup={warmup_steps} ***")
        logger.info(f"Trainable={meta['trainable_params']:,} / Total={meta['total_params']:,} "
                    f"({meta['trainable_pct']*100:.3f}%)")

    # evorank 本脚本暂走标准循环：ES/扩张/剪枝已经在每层模块前向/反向里自然发生（通过 controller 的定期调用需额外代码）
    # 为保持 QA 稳定，这里不嵌入 ES 的 mini-val；evorank 若需完整 ES 循环请走 run_benchmark.py
    global_step = 0
    start_t = time.time()
    stop_flag = False
    for epoch in range(max(1, args.num_train_epochs if args.max_train_steps <= 0 else 10**9)):
        if stop_flag:
            break
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        accum_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            if flatlora_mgr is not None:
                flatlora_mgr.prepare_step(global_step)
                flatlora_mgr.perturb_before_forward()
            outputs = model(**batch)
            loss = outputs.loss
            if method == "sora":
                l1 = sum(p.abs().sum() for n, p in model.named_parameters() if n.endswith(".gate"))
                ng = max(1, sum(p.numel() for n, p in model.named_parameters() if n.endswith(".gate")))
                loss = loss + args.sora_sparse_lambda * l1 / ng

            (loss / args.gradient_accumulation_steps).backward()
            if flatlora_mgr is not None:
                flatlora_mgr.restore_after_backward()
            accum_loss += float(loss.detach())

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                   args.max_grad_norm)
                optimizer.step()
                if sparse_optimizer is not None:
                    sparse_optimizer.step()
                    sparse_sched.step()
                lr_sched.step()
                optimizer.zero_grad(set_to_none=True)
                if sparse_optimizer is not None:
                    sparse_optimizer.zero_grad(set_to_none=True)

                if method == "adalora":
                    inner = unwrap_inner_from_training_model(model)
                    if hasattr(inner, "peft_config") and hasattr(inner, "update_and_allocate"):
                        inner.update_and_allocate(global_step)

                global_step += 1
                if is_main and global_step % 10 == 0:
                    logger.info(f"Epoch {epoch} Step {global_step}/{total_steps} "
                                f"loss={accum_loss/args.gradient_accumulation_steps:.4f} "
                                f"lr={lr_sched.get_last_lr()[0]:.2e}")
                accum_loss = 0.0
                if global_step >= total_steps:
                    stop_flag = True
                    break

    total_train_time = time.time() - start_t

    # 评估（rank0）
    eval_metrics: Dict[str, Any] = {}
    if is_main:
        logger.info("*** Running final evaluation ***")
        inner_for_eval = model.module if hasattr(model, "module") else model
        eval_out = evaluate_qa(
            inner_for_eval, eval_ds, val_ex, data_collator, device,
            batch_size=args.per_device_eval_batch_size,
            version_2_with_negative=args.version_2_with_negative,
        )
        eval_metrics = eval_out["metrics"]
        logger.info(f"Eval metrics: {eval_metrics}")

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
            json.dump({
                "task": args.dataset_name,
                "dataset_config": args.dataset_config_name,
                "backbone": args.model_name_or_path,
                "method": method,
                "seed": args.seed,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "target_modules": meta["target_modules"],
                "trainable_params": meta["trainable_params"],
                "total_params": meta["total_params"],
                "trainable_pct": meta["trainable_pct"],
                "total_train_time_sec": total_train_time,
                "total_steps": total_steps,
                "metrics": eval_metrics,
            }, f, indent=2, ensure_ascii=False)
        with open(os.path.join(args.output_dir, "predictions.json"), "w", encoding="utf-8") as f:
            json.dump(eval_out["predictions"], f, ensure_ascii=False, indent=2)

        # CSV 追加
        if args.export_csv:
            _append_csv(args.export_csv, {
                "task": args.dataset_name,
                "dataset_config": args.dataset_config_name or "",
                "backbone": args.model_name_or_path,
                "method": method,
                "seed": args.seed,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "trainable_params": meta["trainable_params"],
                "trainable_pct": meta["trainable_pct"],
                "exact_match": float(eval_metrics.get("exact_match",
                                                     eval_metrics.get("HasAns_exact", 0.0))),
                "f1": float(eval_metrics.get("f1", eval_metrics.get("HasAns_f1", 0.0))),
                "total_train_time_sec": total_train_time,
                "total_steps": total_steps,
                "artifact_dir": os.path.abspath(args.output_dir),
            })

    return eval_metrics


def _append_csv(path: str, row: Dict[str, Any]) -> None:
    fields = ["task", "dataset_config", "backbone", "method", "seed",
              "lora_rank", "lora_alpha", "trainable_params", "trainable_pct",
              "exact_match", "f1", "total_train_time_sec", "total_steps", "artifact_dir"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


# ---------------------------------------------------------------------------
# 7) CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="microsoft/deberta-v3-base")
    p.add_argument("--method", type=str, required=True,
                   choices=["lora", "lora_kaiming", "pissa", "adalora",
                            "evorank", "sora", "flatlora", "toplora"])
    p.add_argument("--dataset_name", type=str, default="squad",
                   choices=["squad", "squad_v2"])
    p.add_argument("--dataset_config_name", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="artifacts/qa_output")
    p.add_argument("--export_csv", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local_rank", type=int, default=-1)

    # 训练超参
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--max_train_steps", type=int, default=0,
                   help=">0 覆盖 num_train_epochs，用于冒烟")
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true")

    # LoRA / PEFT 结构
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default="",
                   help="留空则根据模型类型自动选择（DeBERTa 默认 6 模块）")

    # 数据
    p.add_argument("--max_seq_length", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--version_2_with_negative", action="store_true",
                   help="SQuAD v2 必开；SQuAD v1 请勿开")
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_eval_samples", type=int, default=0)

    # 算法专属
    p.add_argument("--adalora_tinit", type=int, default=100)
    p.add_argument("--adalora_tfinal", type=int, default=500)
    p.add_argument("--adalora_delta_t", type=int, default=10)
    p.add_argument("--adalora_orth_reg_weight", type=float, default=0.1)
    p.add_argument("--sora_sparse_lambda", type=float, default=1e-3)
    p.add_argument("--flatlora_rho", type=float, default=0.05)
    p.add_argument("--evorank_rho", type=float, default=0.0)
    p.add_argument("--pissa_init_method", type=str, default="pissa_niter_16",
                   choices=["pissa", "pissa_niter_16"])
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    if args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # SQuAD v2 需强制开 version_2_with_negative
    if args.dataset_name == "squad_v2":
        args.version_2_with_negative = True

    run_qa_training(args, args.method)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
