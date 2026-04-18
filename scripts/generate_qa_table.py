"""将 SQuAD v1.1 / v2.0 的 EM/F1 结果汇总为 AdaLoRA 论文 Table 2 风格的表格。

输入
----
``run_qa_benchmark.py`` 以 ``--export_csv`` 追加的 CSV，或 ``artifacts/qa/**/eval_results.json``。
两者均可；CSV 优先，JSON 兜底。

输出
----
一张 Markdown 表格，行=方法，列=预算档位（由 lora_rank 或 trainable_pct 确定），
单元格=``EM / F1`` 或 ``EM±std / F1±std``（多个 seed 聚合）。

Usage
-----
    python scripts/generate_qa_table.py \\
        --csv results_fair_qa_squadv1.csv --task squad \\
        --out_md artifacts/qa/table2_squadv1.md

    python scripts/generate_qa_table.py \\
        --json_glob "artifacts/qa/squadv1_deberta_*/eval_results.json" \\
        --task squad --out_md artifacts/qa/table2_squadv1.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

METHOD_ORDER = ["full", "lora", "lora_kaiming", "pissa", "adalora",
                "evorank", "sora", "flatlora", "toplora"]
METHOD_DISPLAY = {
    "full": "Full FT",
    "lora": "LoRA",
    "lora_kaiming": "LoRA(kaiming)",
    "pissa": "PiSSA",
    "adalora": "AdaLoRA",
    "evorank": "EvoRank",
    "sora": "SoRA",
    "flatlora": "Flat-LoRA",
    "toplora": "TopLoRA",
}


def _to_float(x: Any) -> Optional[float]:
    if x in ("", None, "N/A"):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _fmt_pair(em_mean: Optional[float], em_std: Optional[float],
              f1_mean: Optional[float], f1_std: Optional[float]) -> str:
    def one(m, s):
        if m is None:
            return "-"
        if s is None or s == 0.0 or math.isnan(s):
            return f"{m:.2f}"
        return f"{m:.2f}\u00b1{s:.2f}"
    return f"{one(em_mean, em_std)} / {one(f1_mean, f1_std)}"


def _load_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def _load_jsons(patterns: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    js = json.load(f)
            except Exception:
                continue
            metrics = js.get("metrics", {}) or {}
            rows.append({
                "task": js.get("task", ""),
                "method": js.get("method", ""),
                "seed": str(js.get("seed", "")),
                "lora_rank": str(js.get("lora_rank", "")),
                "trainable_params": str(js.get("trainable_params", "")),
                "trainable_pct": str(js.get("trainable_pct", "")),
                "exact_match": str(metrics.get("exact_match",
                                               metrics.get("HasAns_exact", ""))),
                "f1": str(metrics.get("f1",
                                      metrics.get("HasAns_f1", ""))),
            })
    return rows


def _aggregate(rows: List[Dict[str, str]], task_filter: Optional[str]
               ) -> Tuple[Dict[Tuple[str, str], Dict[str, Optional[float]]], List[Tuple[str, float]]]:
    """返回:
    - results[(method, budget_tag)] = {'em_mean','em_std','f1_mean','f1_std','n'}
    - columns: [(budget_tag, trainable_pct_mean), ...] 升序
    """
    by_bucket: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: {"em": [], "f1": [], "pct": []})
    for r in rows:
        task = (r.get("task") or "").strip().lower()
        if task_filter and task != task_filter.lower():
            continue
        method = (r.get("method") or "").strip().lower()
        rank = (r.get("lora_rank") or "").strip()
        pct = _to_float(r.get("trainable_pct"))
        em = _to_float(r.get("exact_match"))
        f1 = _to_float(r.get("f1"))
        if not method or em is None or f1 is None:
            continue
        # budget_tag 以 rank 为主；否则以 pct 精度为 tag
        if rank:
            tag = f"r={rank}"
        elif pct is not None:
            tag = f"{pct*100:.2f}%"
        else:
            tag = "?"
        by_bucket[(method, tag)]["em"].append(em)
        by_bucket[(method, tag)]["f1"].append(f1)
        if pct is not None:
            by_bucket[(method, tag)]["pct"].append(pct)

    def _ms(v):
        if not v:
            return None, None
        mean = sum(v) / len(v)
        if len(v) < 2:
            return mean, None
        var = sum((x - mean) ** 2 for x in v) / (len(v) - 1)
        return mean, math.sqrt(var)

    results: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
    budget_pct: Dict[str, List[float]] = defaultdict(list)
    for (method, tag), store in by_bucket.items():
        em_m, em_s = _ms(store["em"])
        f1_m, f1_s = _ms(store["f1"])
        results[(method, tag)] = {
            "em_mean": em_m, "em_std": em_s,
            "f1_mean": f1_m, "f1_std": f1_s,
            "n": float(len(store["em"])),
        }
        if store["pct"]:
            budget_pct[tag].append(sum(store["pct"]) / len(store["pct"]))

    def _tag_key(tag: str) -> float:
        if tag.startswith("r="):
            try:
                return float(tag[2:])
            except Exception:
                return float("inf")
        if tag.endswith("%"):
            try:
                return float(tag[:-1])
            except Exception:
                return float("inf")
        return float("inf")

    columns = sorted(
        ((t, (sum(pcts) / len(pcts)) if pcts else float("nan"))
         for t, pcts in budget_pct.items()),
        key=lambda x: _tag_key(x[0]),
    )
    # 有些行可能没有 pct，但仍有 tag；把这些 tag 补齐
    for tag in {k[1] for k in results.keys()} - {c[0] for c in columns}:
        columns.append((tag, float("nan")))
    columns.sort(key=lambda x: _tag_key(x[0]))
    return results, columns


def _render_markdown(results, columns, title: str) -> str:
    methods_present = sorted({k[0] for k in results.keys()},
                             key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    header1 = ["Method"] + [f"{t} (~{p*100:.2f}%)" if not math.isnan(p) else t for t, p in columns]
    sep = ["---"] * len(header1)
    lines = [f"### {title}", "",
             "| " + " | ".join(header1) + " |",
             "| " + " | ".join(sep) + " |"]
    for m in methods_present:
        row = [METHOD_DISPLAY.get(m, m)]
        for tag, _ in columns:
            cell = results.get((m, tag))
            if cell is None:
                row.append("-")
            else:
                row.append(_fmt_pair(cell["em_mean"], cell["em_std"],
                                     cell["f1_mean"], cell["f1_std"]))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("> 每格格式: `EM / F1`（多 seed 时为 `mean±std`）。")
    lines.append("> 单元格缺失表示该方法×预算尚未跑通或未导出。")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=None, help="结果 CSV（来自 run_qa_benchmark 的 --export_csv）")
    p.add_argument("--json_glob", type=str, action="append", default=[],
                   help="eval_results.json 的 glob 模式，可多次提供")
    p.add_argument("--task", type=str, required=True,
                   choices=["squad", "squad_v2"],
                   help="只聚合指定任务的行")
    p.add_argument("--title", type=str, default=None,
                   help="Markdown 标题，默认据 task 生成")
    p.add_argument("--out_md", type=str, default=None, help="输出 markdown 路径；缺省仅打印")
    args = p.parse_args()

    rows: List[Dict[str, str]] = []
    if args.csv:
        rows.extend(_load_csv(args.csv))
    if args.json_glob:
        rows.extend(_load_jsons(args.json_glob))
    if not rows:
        raise SystemExit("未找到任何结果行，请检查 --csv / --json_glob。")

    results, columns = _aggregate(rows, task_filter=args.task)
    if not results:
        raise SystemExit(f"过滤后无数据：task={args.task}")

    title = args.title or (
        "Table 2 — SQuAD v1.1 (DeBERTa-v3-base, EM / F1)"
        if args.task == "squad" else
        "Table 2 — SQuAD v2.0 (DeBERTa-v3-base, EM / F1)"
    )
    md = _render_markdown(results, columns, title)
    print(md)
    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)) or ".", exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"\nWrote {args.out_md}")


if __name__ == "__main__":
    main()
