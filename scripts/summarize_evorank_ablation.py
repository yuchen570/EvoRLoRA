import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]

ABLATION_ORDER = [
    "full",
    "no_complexity",
    "zero_init",
    "no_ema",
    "no_persist_only",
    "no_cooldown_only",
    "no_persist_cooldown",
    "no_reallocation",
    "no_noop",
    "es_budget_light",
    "es_budget_heavy",
]

TASK_ORDER = ["mnli", "sst2", "cola", "qqp", "qnli", "mrpc", "stsb"]

TASK_HEADERS = {
    "mnli": "MNLI (m/mm)",
    "sst2": "SST-2",
    "cola": "CoLA",
    "qqp": "QQP (Acc/F1)",
    "qnli": "QNLI",
    "mrpc": "MRPC (Acc/F1)",
    "stsb": "STS-B",
}

VARIANT_LABELS = {
    "full": "Full",
    "no_complexity": "w/o Complexity Reward",
    "zero_init": "Zero Init",
    "no_ema": "w/o EMA",
    "no_persist_only": "w/o Persistence Threshold",
    "no_cooldown_only": "w/o Cooldown",
    "no_persist_cooldown": "w/o Persistence+Cooldown",
    "no_reallocation": "w/o Reallocation",
    "no_noop": "w/o No-op Candidate",
    "es_budget_light": "ES Budget Light",
    "es_budget_heavy": "ES Budget Heavy",
}


def _safe_float(value: object) -> Optional[float]:
    if value in ("", None, "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_rows(csv_path: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    mean_rows: Dict[str, Dict[str, str]] = {}
    std_rows: Dict[str, Dict[str, str]] = {}
    raw_rows: Dict[str, Dict[str, str]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row.get("task", "")
            seed = row.get("seed", "")
            if not task:
                continue
            if seed == "mean":
                mean_rows[task] = row
            elif seed == "std":
                std_rows[task] = row
            elif task not in raw_rows:
                raw_rows[task] = row

    if not mean_rows:
        mean_rows = raw_rows
    return mean_rows, std_rows


def _task_score(task: str, row: Dict[str, str]) -> Optional[float]:
    if task == "mnli":
        acc_m = _safe_float(row.get("accuracy_m"))
        acc_mm = _safe_float(row.get("accuracy_mm"))
        if acc_m is None and acc_mm is None:
            return _safe_float(row.get("accuracy"))
        vals = [v for v in (acc_m, acc_mm) if v is not None]
        return sum(vals) / len(vals) if vals else None
    if task == "sst2":
        return _safe_float(row.get("accuracy"))
    if task == "cola":
        return _safe_float(row.get("matthews_corrcoef"))
    if task == "qqp":
        acc = _safe_float(row.get("accuracy"))
        f1 = _safe_float(row.get("f1"))
        vals = [v for v in (acc, f1) if v is not None]
        return sum(vals) / len(vals) if vals else None
    if task == "qnli":
        return _safe_float(row.get("accuracy"))
    if task == "mrpc":
        acc = _safe_float(row.get("accuracy"))
        f1 = _safe_float(row.get("f1"))
        vals = [v for v in (acc, f1) if v is not None]
        return sum(vals) / len(vals) if vals else None
    if task == "stsb":
        ps = _safe_float(row.get("pearson_spearman_mean"))
        if ps is not None:
            return ps
        pearson = _safe_float(row.get("pearson"))
        spearman = _safe_float(row.get("spearman"))
        vals = [v for v in (pearson, spearman) if v is not None]
        return sum(vals) / len(vals) if vals else None
    return None


def _task_display(task: str, mean_row: Dict[str, str], std_row: Optional[Dict[str, str]]) -> str:
    def fmt(mean_v: Optional[float], std_v: Optional[float]) -> str:
        if mean_v is None:
            return "-"
        mean_pct = mean_v * 100.0
        if std_v is None:
            return f"{mean_pct:.2f}"
        return f"{mean_pct:.2f} ± {std_v * 100.0:.2f}"

    if task == "mnli":
        m = _safe_float(mean_row.get("accuracy_m"))
        mm = _safe_float(mean_row.get("accuracy_mm"))
        sm = _safe_float(std_row.get("accuracy_m")) if std_row else None
        smm = _safe_float(std_row.get("accuracy_mm")) if std_row else None
        if m is None and mm is None:
            return fmt(_safe_float(mean_row.get("accuracy")), _safe_float(std_row.get("accuracy")) if std_row else None)
        return f"{fmt(m, sm)}/{fmt(mm, smm)}"
    if task in {"qqp", "mrpc"}:
        acc = _safe_float(mean_row.get("accuracy"))
        f1 = _safe_float(mean_row.get("f1"))
        sacc = _safe_float(std_row.get("accuracy")) if std_row else None
        sf1 = _safe_float(std_row.get("f1")) if std_row else None
        return f"{fmt(acc, sacc)}/{fmt(f1, sf1)}"
    metric_map = {
        "sst2": "accuracy",
        "cola": "matthews_corrcoef",
        "qnli": "accuracy",
        "stsb": "pearson_spearman_mean",
    }
    key = metric_map[task]
    return fmt(_safe_float(mean_row.get(key)), _safe_float(std_row.get(key)) if std_row else None)


def _collect_variant_row(variant: str) -> Optional[Dict[str, object]]:
    csv_path = ROOT / f"results_evorank_{variant}.csv"
    if not csv_path.exists():
        return None

    mean_rows, std_rows = _pick_rows(csv_path)
    row: Dict[str, object] = {
        "variant_key": variant,
        "variant": VARIANT_LABELS.get(variant, variant),
        "source_csv": str(csv_path),
    }

    scores: List[float] = []
    for task in TASK_ORDER:
        mean_row = mean_rows.get(task)
        std_row = std_rows.get(task)
        if not mean_row:
            row[task] = "-"
            row[f"{task}_score"] = None
            continue
        row[task] = _task_display(task, mean_row, std_row)
        score = _task_score(task, mean_row)
        row[f"{task}_score"] = score
        if score is not None:
            scores.append(score)

    row["all_avg"] = sum(scores) / len(scores) if scores else None
    params = {_safe_float(mean_rows[t].get("trainable_params")) for t in mean_rows if mean_rows[t].get("trainable_params", "")}
    params = {p for p in params if p is not None}
    row["trainable_params"] = int(next(iter(params))) if params else ""
    return row


def _write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    fieldnames = ["variant", "trainable_params"] + TASK_ORDER + ["all_avg_pct", "delta_vs_full_pct", "source_csv"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {
                "variant": row["variant"],
                "trainable_params": row["trainable_params"],
                "source_csv": row["source_csv"],
            }
            for task in TASK_ORDER:
                out[task] = row[task]
            all_avg = row.get("all_avg")
            delta = row.get("delta_vs_full")
            out["all_avg_pct"] = f"{all_avg * 100.0:.2f}" if isinstance(all_avg, float) else ""
            out["delta_vs_full_pct"] = f"{delta * 100.0:+.2f}" if isinstance(delta, float) else ""
            writer.writerow(out)


def _write_markdown(rows: List[Dict[str, object]], out_path: Path) -> None:
    headers = ["Variant", "# Params"] + [TASK_HEADERS[t] for t in TASK_ORDER] + ["All Avg", "Δ vs Full"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        params = row["trainable_params"]
        params_str = f"{int(params) / 1_000_000:.2f}M" if isinstance(params, int) and params > 0 else "-"
        all_avg = row.get("all_avg")
        delta = row.get("delta_vs_full")
        all_avg_str = f"{all_avg * 100.0:.2f}" if isinstance(all_avg, float) else "-"
        delta_str = f"{delta * 100.0:+.2f}" if isinstance(delta, float) else "-"
        values = [row["variant"], params_str] + [str(row[t]) for t in TASK_ORDER] + [all_avg_str, delta_str]
        lines.append("| " + " | ".join(values) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows: List[Dict[str, object]] = []
    for variant in ABLATION_ORDER:
        row = _collect_variant_row(variant)
        if row is not None:
            rows.append(row)

    if not rows:
        print("未找到任何 EvoRank 主消融结果 CSV。请先运行 scripts/ablate_evorank_glue_deberta.sh")
        return

    full_avg = next((row["all_avg"] for row in rows if row["variant_key"] == "full"), None)
    for row in rows:
        if isinstance(full_avg, float) and isinstance(row.get("all_avg"), float):
            row["delta_vs_full"] = row["all_avg"] - full_avg
        else:
            row["delta_vs_full"] = None

    csv_out = ROOT / "results_evorank_ablation_summary.csv"
    md_out = ROOT / "results_evorank_ablation_summary.md"
    _write_csv(rows, csv_out)
    _write_markdown(rows, md_out)

    print(f"Wrote CSV summary to {csv_out}")
    print(f"Wrote Markdown summary to {md_out}")


if __name__ == "__main__":
    main()
