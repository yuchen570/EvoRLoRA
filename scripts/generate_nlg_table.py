"""将 NLG（XSum/CNN-DailyMail）ROUGE 结果汇总为 AdaLoRA 论文 Table 3 风格的表格。

读取由 ``run_benchmark.py`` 在 NLG 任务上导出的 CSV（每行已包含 rouge1/2/L），
按 (method, trainable_params) 聚合，并按任务一行 ``R-1/R-2/R-L`` 展示；
若存在 ``seed='mean'`` / ``seed='std'`` 行则显示 ``mean±std``，否则退化为单点。

Usage
-----
    python scripts/generate_nlg_table.py \
        --xsum_csv results_fair_nlg_xsum_ddp.csv \
        --cnndm_csv results_fair_nlg_cnndm_ddp.csv \
        --out_md artifacts/nlg/table3_nlg_rouge.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

METHOD_ORDER = ["full", "lora", "adalora", "pissa", "evorank",
                "sora", "flatlora", "toplora", "lora_kaiming"]

METHOD_DISPLAY = {
    "full": "Full FT",
    "lora": "LoRA",
    "adalora": "AdaLoRA",
    "pissa": "PiSSA",
    "evorank": "EvoRank",
    "sora": "SoRA",
    "flatlora": "Flat-LoRA",
    "toplora": "TopLoRA",
    "lora_kaiming": "LoRA(kaiming)",
}


def _f(x) -> Optional[float]:
    if x in ("", None, "N/A"):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _fmt(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "-"
    m = mean * 100.0
    if std is None or std == 0.0:
        return f"{m:.2f}"
    s = std * 100.0
    return f"{m:.2f}\u00b1{s:.2f}"


def _aggregate_from_rows(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, Tuple[Optional[float], Optional[float]]]]:
    """按 (method, task) 聚合 ROUGE 指标。

    优先使用 CSV 中已有的 ``seed='mean'``/``seed='std'`` 行；
    若缺失则从普通数据行重新计算 mean/std。
    """
    mean_rows: Dict[Tuple[str, str], Dict[str, str]] = {}
    std_rows: Dict[Tuple[str, str], Dict[str, str]] = {}
    seed_rows: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    params: Dict[str, str] = {}

    for row in rows:
        method = row.get("method", "")
        task = row.get("task", "")
        seed = str(row.get("seed", ""))
        key = (method, task)
        if seed == "mean":
            mean_rows[key] = row
        elif seed == "std":
            std_rows[key] = row
        else:
            seed_rows[key].append(row)
        if method and method not in params:
            tp = row.get("trainable_params", "")
            if tp:
                try:
                    params[method] = f"{int(tp) / 1_000_000:.2f}M"
                except Exception:
                    params[method] = tp

    agg: Dict[Tuple[str, str], Dict[str, Tuple[Optional[float], Optional[float]]]] = defaultdict(dict)
    for key in set(list(mean_rows.keys()) + list(seed_rows.keys())):
        for metric in ("rouge1", "rouge2", "rougeL"):
            mean: Optional[float] = None
            std: Optional[float] = None
            if key in mean_rows:
                mean = _f(mean_rows[key].get(metric))
                if key in std_rows:
                    std = _f(std_rows[key].get(metric))
            if mean is None and key in seed_rows:
                vals = [v for v in (_f(r.get(metric)) for r in seed_rows[key]) if v is not None]
                if vals:
                    mean = sum(vals) / len(vals)
                    if len(vals) > 1:
                        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                        std = math.sqrt(max(var, 0.0))
                    else:
                        std = None
            agg[key][metric] = (mean, std)
    return agg, params


def _load_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _format_section(title: str, task_hint: str, agg, params: Dict[str, str]) -> str:
    """渲染单个任务的 ROUGE 表（行：method；列：R-1/R-2/R-L）。"""
    # 找这一张 CSV 下所有 method（不同 task 名可能被简化成 task_hint 或完整 nlg_dataset_name）
    methods = sorted({m for (m, t) in agg.keys()})
    ordered = [m for m in METHOD_ORDER if m in methods] + [m for m in methods if m not in METHOD_ORDER]

    lines = [f"### {title}", "", "| Method | # Params | R-1 | R-2 | R-L |", "|---|---|---|---|---|"]
    for method in ordered:
        row = {}
        for (m, t), metrics in agg.items():
            if m != method:
                continue
            # 匹配 task：xsum / cnn_dailymail / task_hint 任一即可
            if task_hint and t not in (task_hint, title.lower(), title):
                # 允许 "xsum" vs "XSum" 的差异
                if t.lower() != task_hint.lower():
                    continue
            row = metrics
            break
        if not row:
            continue
        r1 = _fmt(*row.get("rouge1", (None, None)))
        r2 = _fmt(*row.get("rouge2", (None, None)))
        rL = _fmt(*row.get("rougeL", (None, None)))
        lines.append(
            f"| {METHOD_DISPLAY.get(method, method)} | {params.get(method, '-')} | {r1} | {r2} | {rL} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xsum_csv", default="results_fair_nlg_xsum_ddp.csv")
    ap.add_argument("--cnndm_csv", default="results_fair_nlg_cnndm_ddp.csv")
    ap.add_argument("--extra_csvs", nargs="*", default=[],
                    help="附加 CSV（如 ablation），会自动按 task 分组展示。")
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    sections: List[str] = []
    for title, hint, path in [("XSum", "xsum", args.xsum_csv),
                              ("CNN/DailyMail", "cnn_dailymail", args.cnndm_csv)]:
        rows = _load_csv(path)
        if not rows:
            sections.append(f"### {title}\n\n_CSV `{path}` 不存在或为空——请先运行对应训练脚本。_\n")
            continue
        agg, params = _aggregate_from_rows(rows)
        sections.append(_format_section(title, hint, agg, params))

    # extra CSVs
    for path in args.extra_csvs:
        for p in glob.glob(path):
            rows = _load_csv(p)
            if not rows:
                continue
            agg, params = _aggregate_from_rows(rows)
            # 按 CSV 里出现的 task 分别渲染
            tasks = sorted({t for (_, t) in agg.keys()})
            for t in tasks:
                sections.append(_format_section(t, t, agg, params))

    md = "# NLG Results (AdaLoRA-style Table 3)\n\n_We report R-1 / R-2 / R-L (percentages). Mean\u00b1std across seeds when available._\n\n" + "\n".join(sections)

    print(md)
    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\nSaved to {args.out_md}")


if __name__ == "__main__":
    main()
