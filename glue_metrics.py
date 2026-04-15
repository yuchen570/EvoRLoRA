"""GLUE 各子集 NLU 指标与选优标量。

论文表口径：MNLI 为 matched/mismatched 准确率；SST-2 / QNLI / RTE / WNLI 为 Acc；
CoLA 为 Matthews；QQP 为 Acc 与 F1 的**平均**作为选优标准；MRPC 为 Acc；
STS-B 为 Pearson 与 Spearman 的均值（Corr）。"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def glue_primary_metric_key(task_name: str) -> str:
    """用于日志 / TensorBoard / metrics.jsonl / CSV「主指标」列名的标识（与选优标量一致）。"""
    m = {
        "cola": "matthews_corrcoef",
        "sst2": "accuracy",
        "mrpc": "accuracy",
        "qqp": "mean_acc_f1",
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
    返回该子集的选优标量（越大越好，用于 best 跟踪、epoch 日志与 CSV 主列）。
    - CoLA: Matthews
    - SST-2 / MNLI / QNLI / RTE / WNLI / ax / MRPC: Accuracy
    - QQP: (Accuracy + F1) / 2
    - STS-B: (Pearson + Spearman) / 2
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if task_name == "cola":
        return float(matthews_corrcoef(y_true, y_pred))
    if task_name in ("sst2", "mnli", "qnli", "rte", "wnli", "ax", "mrpc"):
        return float(accuracy_score(y_true, y_pred))
    if task_name == "qqp":
        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
        return (acc + f1) / 2.0
    if task_name == "stsb":
        p, _ = pearsonr(y_pred, y_true)
        s, _ = spearmanr(y_pred, y_true)
        p = 0.0 if np.isnan(p) else float(p)
        s = 0.0 if np.isnan(s) else float(s)
        return (p + s) / 2.0
    raise ValueError(f"未知 GLUE task_name: {task_name}")

def compute_glue_metrics_dict(task_name: str, y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """返回该子集的所有评估指标字典，用于记录和存放到 CSV 结果中。"""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    res = {}
    if task_name == "cola":
        res["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
    elif task_name in ("sst2", "mnli", "qnli", "rte", "wnli", "ax"):
        res["accuracy"] = float(accuracy_score(y_true, y_pred))
    elif task_name in ("mrpc", "qqp"):
        res["f1"] = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
        res["accuracy"] = float(accuracy_score(y_true, y_pred))
    elif task_name == "stsb":
        p, _ = pearsonr(y_pred, y_true)
        s, _ = spearmanr(y_pred, y_true)
        p = 0.0 if np.isnan(p) else float(p)
        s = 0.0 if np.isnan(s) else float(s)
        res["pearson_spearman_mean"] = (p + s) / 2.0
        res["pearson"] = p
        res["spearman"] = s
    else:
        raise ValueError(f"未知 GLUE task_name: {task_name}")
    return res


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
