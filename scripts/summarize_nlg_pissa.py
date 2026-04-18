"""扫描 ``artifacts/nlg/**/eval_results.json``，打印 PiSSA 论文 Table 2 风格的汇总表格。

列: GSM8K(acc) | MATH(acc) | HumanEval(pass@1) | MBPP(pass@1) | MT-Bench(未评则 '-')
行: (Model, Strategy) 聚合到 mean±std（跨 seed）

Usage
-----
    python scripts/summarize_nlg_pissa.py \
        --root artifacts/nlg \
        --out_md artifacts/nlg/table2_nlg_summary.md

默认也会在标准输出上打印一份。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


STRATEGY_DISPLAY = {
    "lora": "LoRA(gaussian)",
    "lora_kaiming": "LoRA(kaiming)",
    "pissa": "PiSSA",
    "adalora": "AdaLoRA",
    "evorank": "EvoRank",
    "sora": "SoRA",
    "flatlora": "Flat-LoRA",
    "toplora": "TopLoRA",
    "full": "Full FT",
}

METHOD_ORDER = ["full", "lora", "lora_kaiming", "pissa", "adalora",
                "evorank", "sora", "flatlora", "toplora"]

COLUMNS = [
    ("GSM8K", "gsm8k", "acc"),
    ("MATH", "math", "acc"),
    ("HumanEval", "humaneval", "pass@1_base"),
    ("MBPP", "mbpp", "pass@1_base"),
    ("MT-Bench", "mt_bench", "score"),
]


def _pct(x: float) -> float:
    # 跟 PiSSA 表对齐：accuracy / pass@1 以百分比显示；MT-Bench 直接用分值
    return x * 100.0


def _fmt_cell(vals: List[float], *, percent: bool) -> str:
    if not vals:
        return "-"
    if percent:
        vals = [_pct(v) for v in vals]
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return f"{mean:.2f}"
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    std = math.sqrt(max(var, 0.0))
    return f"{mean:.2f}\u00b1{std:.2f}"


def _load_records(root: str) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for p in glob.glob(os.path.join(root, "**", "eval_results.json"), recursive=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                recs.append(json.load(f))
        except Exception:
            continue
    return recs


def _infer_model_tag(rec: Dict[str, Any]) -> str:
    tag = rec.get("model_tag", "").strip()
    if tag and tag != "unknown":
        return tag
    # 尝试从路径推断（eval_results.json 同级 benchmark_meta.json 可能有更多信息）
    return tag or "unknown"


def build_rows(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, List[float]]]:
    """返回 {(model, method): {col_key: [value_per_seed]}}"""
    buckets: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        model = _infer_model_tag(rec)
        method = rec.get("method", "unknown")
        metrics = rec.get("metrics", {}) or {}
        for _, task_key, val_key in COLUMNS:
            m = metrics.get(task_key)
            if not isinstance(m, dict):
                continue
            # HumanEval/MBPP 可能只记录了 pass@1（无 base/plus 前缀）
            v: Optional[float] = m.get(val_key)
            if v is None and task_key in ("humaneval", "mbpp"):
                v = m.get("pass@1") or m.get("pass@1_plus")
            if v is None:
                continue
            try:
                buckets[(model, method)][task_key].append(float(v))
            except Exception:
                pass
    return buckets


def render_markdown(buckets: Dict[Tuple[str, str], Dict[str, List[float]]]) -> str:
    models = sorted({m for m, _ in buckets.keys()})
    header = "| Model | Strategy | " + " | ".join(c[0] for c in COLUMNS) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(COLUMNS)) + "|"
    lines = [header, sep]

    for model in models:
        methods_in_model = [m for (mm, m) in buckets.keys() if mm == model]
        ordered = [m for m in METHOD_ORDER if m in methods_in_model] + \
                  [m for m in methods_in_model if m not in METHOD_ORDER]
        for i, method in enumerate(ordered):
            row = buckets[(model, method)]
            cells = []
            for _, task_key, _ in COLUMNS:
                vals = row.get(task_key, [])
                percent = task_key in ("gsm8k", "math", "humaneval", "mbpp")
                cells.append(_fmt_cell(vals, percent=percent))
            model_col = model if i == 0 else ""
            strategy = STRATEGY_DISPLAY.get(method, method)
            lines.append(f"| {model_col} | {strategy} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/nlg", help="搜索 eval_results.json 的根目录")
    ap.add_argument("--out_md", default=None, help="若提供，则把 Markdown 写入该文件")
    args = ap.parse_args()

    records = _load_records(args.root)
    if not records:
        print(f"[warn] {args.root} 下未找到任何 eval_results.json；请先运行 scripts/eval_nlg_pissa.py")
        return

    buckets = build_rows(records)
    md = render_markdown(buckets)
    print("\nTable 2 (PiSSA) -- NLG task comparison (higher is better)")
    print(md)
    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("# NLG Results (PiSSA-style Table 2)\n\n")
            f.write(md)
            f.write("\n")
        print(f"\nSaved Markdown to {args.out_md}")


if __name__ == "__main__":
    main()
