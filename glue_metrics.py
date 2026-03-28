"""GLUE 各子集官方主指标（与 GLUE benchmark 常用约定一致）。"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def glue_primary_metric_key(task_name: str) -> str:
    """用于日志 / TensorBoard / metrics.jsonl 的指标标识。"""
    m = {
        "cola": "matthews_corrcoef",
        "sst2": "accuracy",
        "mrpc": "f1",
        "qqp": "f1",
        "stsb": "pearson_spearman_mean",
        "mnli": "accuracy",
        "qnli": "accuracy",
        "rte": "accuracy",
        "wnli": "accuracy",
        "ax": "accuracy",
    }
    if task_name not in m:
        raise ValueError(f"未知 GLUE task_name: {task_name}")
    return m[task_name]


def compute_glue_primary_metric(task_name: str, y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    返回该子集的「主」标量（越大越好，用于 best 跟踪与 CSV）。
    - CoLA: Matthews 相关
    - SST-2 / MNLI / QNLI / RTE / WNLI / ax: Accuracy
    - MRPC / QQP: F1 (binary)
    - STS-B: (Pearson + Spearman) / 2（与 GLUE 总分中该任务常见合成方式一致）
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if task_name == "cola":
        return float(matthews_corrcoef(y_true, y_pred))
    if task_name in ("sst2", "mnli", "qnli", "rte", "wnli", "ax"):
        return float(accuracy_score(y_true, y_pred))
    if task_name in ("mrpc", "qqp"):
        return float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    if task_name == "stsb":
        p, _ = pearsonr(y_pred, y_true)
        s, _ = spearmanr(y_pred, y_true)
        p = 0.0 if np.isnan(p) else float(p)
        s = 0.0 if np.isnan(s) else float(s)
        return (p + s) / 2.0
    raise ValueError(f"未知 GLUE task_name: {task_name}")


@torch.no_grad()
def collect_nlu_predictions(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    regression: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """在给定 loader 上收集整集预测与标签（用于主指标）。"""
    preds_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    model.eval()
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels_b = batch["labels"]
        feats = {k: v for k, v in batch.items() if k != "labels"}
        logits = model(feats)
        if regression:
            preds_chunks.append(logits.squeeze(-1).detach().float().cpu().numpy())
            labels_chunks.append(labels_b.detach().float().cpu().numpy())
        else:
            preds_chunks.append(logits.argmax(dim=-1).detach().cpu().numpy())
            labels_chunks.append(labels_b.detach().cpu().numpy())
    if not preds_chunks:
        return np.array([]), np.array([])
    return np.concatenate(preds_chunks, axis=0), np.concatenate(labels_chunks, axis=0)
