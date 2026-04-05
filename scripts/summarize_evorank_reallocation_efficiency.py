import csv
from pathlib import Path
from typing import Dict, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ["capped_k8", "unlimited"]
LABELS = {
    "capped_k8": "Top-k Cross (K=8)",
    "unlimited": "Unlimited Reallocation",
}


def _safe_float(value: object) -> Optional[float]:
    if value in ("", None, "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_best_rows(csv_path: Path) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, str]]]:
    mean_row = None
    std_row = None
    raw_row = None
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seed = row.get("seed", "")
            if seed == "mean":
                mean_row = row
            elif seed == "std":
                std_row = row
            elif raw_row is None:
                raw_row = row
    return mean_row or raw_row, std_row


def _metric_display(mean_row: Dict[str, str], std_row: Optional[Dict[str, str]]) -> Tuple[str, Optional[float]]:
    key = mean_row.get("val_metric_key", "")
    if key == "accuracy":
        mean_v = _safe_float(mean_row.get("accuracy"))
        std_v = _safe_float(std_row.get("accuracy")) if std_row else None
    elif key == "matthews_corrcoef":
        mean_v = _safe_float(mean_row.get("matthews_corrcoef"))
        std_v = _safe_float(std_row.get("matthews_corrcoef")) if std_row else None
    elif key == "pearson_spearman_mean":
        mean_v = _safe_float(mean_row.get("pearson_spearman_mean"))
        std_v = _safe_float(std_row.get("pearson_spearman_mean")) if std_row else None
    else:
        mean_v = _safe_float(mean_row.get("accuracy"))
        std_v = _safe_float(std_row.get("accuracy")) if std_row else None
    if mean_v is None:
        return "-", None
    if std_v is None:
        return f"{mean_v * 100.0:.2f}", mean_v
    return f"{mean_v * 100.0:.2f} ± {std_v * 100.0:.2f}", mean_v


def main() -> None:
    rows = []
    base_time = None

    for variant in VARIANTS:
        csv_path = ROOT / f"results_evorank_reallocation_{variant}.csv"
        if not csv_path.exists():
            continue
        mean_row, std_row = _pick_best_rows(csv_path)
        if mean_row is None:
            continue
        metric_display, metric_value = _metric_display(mean_row, std_row)
        total_time = _safe_float(mean_row.get("total_train_time_sec"))
        peak_mem = _safe_float(mean_row.get("peak_memory_mb"))
        avg_rank = _safe_float(mean_row.get("avg_active_rank"))
        if variant == "capped_k8" and total_time is not None:
            base_time = total_time
        rows.append(
            {
                "variant": LABELS.get(variant, variant),
                "task": mean_row.get("task", ""),
                "val_metric_key": mean_row.get("val_metric_key", ""),
                "val_metric": metric_display,
                "avg_active_rank": f"{avg_rank:.2f}" if avg_rank is not None else "-",
                "peak_memory_mb": f"{peak_mem:.2f}" if peak_mem is not None else "-",
                "total_train_time_sec": f"{total_time:.2f}" if total_time is not None else "-",
                "relative_time_vs_k8": "",
                "source_csv": str(csv_path),
            }
        )

    if not rows:
        print("未找到任何 Reallocation 效率消融结果 CSV。请先运行 scripts/ablate_evorank_reallocation_efficiency.sh")
        return

    for row in rows:
        try:
            total_time = float(row["total_train_time_sec"])
        except (TypeError, ValueError):
            total_time = None
        if base_time is not None and total_time is not None and base_time > 0:
            row["relative_time_vs_k8"] = f"{total_time / base_time:.2f}x"
        else:
            row["relative_time_vs_k8"] = "-"

    csv_out = ROOT / "results_evorank_reallocation_efficiency_summary.csv"
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "task",
                "val_metric_key",
                "val_metric",
                "avg_active_rank",
                "peak_memory_mb",
                "total_train_time_sec",
                "relative_time_vs_k8",
                "source_csv",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines = [
        "| Variant | Task | Val Metric | Avg Active Rank | Peak Memory (MB) | Train Time (s) | Relative Time vs K=8 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['variant']} | {row['task']} | {row['val_metric']} | {row['avg_active_rank']} | "
            f"{row['peak_memory_mb']} | {row['total_train_time_sec']} | {row['relative_time_vs_k8']} |"
        )
    md_out = ROOT / "results_evorank_reallocation_efficiency_summary.md"
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote CSV summary to {csv_out}")
    print(f"Wrote Markdown summary to {md_out}")


if __name__ == "__main__":
    main()
