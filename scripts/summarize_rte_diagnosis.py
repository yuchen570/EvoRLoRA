import argparse
import csv
import datetime as dt
import math
import re
from pathlib import Path
from typing import Dict, List, Optional


EPOCH_RE = re.compile(
    r"^\[(?P<method>[^\]]+)\]\s+epoch=(?P<epoch>\d+)/(?P<epochs>\d+)\s+step=(?P<step>\d+)\s+"
    r"val_(?P<metric>[a-zA-Z0-9_]+)=(?P<val>[-+0-9.eE]+)\s+best=(?P<best>[-+0-9.eE]+)"
)
RANK_HEADER_RE = re.compile(r"^\[(?P<method>[^\]]+)\]\s+=== Rank Distribution")
RANK_RE = re.compile(r"avg_rank=(?P<avg>[-+0-9.eE]+)\s+total_active=(?P<active>\d+)/(?P<cap>\d+)")
EVORANK_ES_RE = re.compile(
    r"^\[evorank\]\[es\]\s+step=(?P<step>\d+).+delta_val_loss=(?P<dval>[-+0-9.eE]+)\s+"
    r"delta_complexity=(?P<dcomp>[-+0-9.eE]+)"
)


def _safe_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    try:
        v = float(text)
    except ValueError:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def parse_rte_log(log_path: Path) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    if not log_path.exists():
        return summary

    current_rank_method: Optional[str] = None

    def ensure(method: str) -> Dict[str, object]:
        if method not in summary:
            summary[method] = {
                "method": method,
                "metric_key": "",
                "best_val": None,
                "final_val": None,
                "rank_start": None,
                "rank_end": None,
                "rank_delta": None,
                "num_eval_points": 0,
                "stuck_majority_band": "",
                "_val_hist": [],
                "_evorank_dval": [],
                "_evorank_dcomp": [],
            }
        return summary[method]

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            m_epoch = EPOCH_RE.match(line)
            if m_epoch:
                method = m_epoch.group("method")
                rec = ensure(method)
                metric = m_epoch.group("metric")
                val = _safe_float(m_epoch.group("val"))
                best = _safe_float(m_epoch.group("best"))
                rec["metric_key"] = metric
                if val is not None:
                    rec["final_val"] = val
                    rec["_val_hist"].append(val)
                    rec["num_eval_points"] = int(rec["num_eval_points"]) + 1
                if best is not None:
                    rec["best_val"] = best if rec["best_val"] is None else max(float(rec["best_val"]), best)
                continue

            m_header = RANK_HEADER_RE.match(line)
            if m_header:
                current_rank_method = m_header.group("method")
                ensure(current_rank_method)
                continue

            m_rank = RANK_RE.search(line)
            if m_rank and current_rank_method is not None:
                rec = ensure(current_rank_method)
                avg_rank = _safe_float(m_rank.group("avg"))
                if avg_rank is not None:
                    if rec["rank_start"] is None:
                        rec["rank_start"] = avg_rank
                    rec["rank_end"] = avg_rank
                continue

            m_es = EVORANK_ES_RE.match(line)
            if m_es:
                rec = ensure("evorank")
                dval = _safe_float(m_es.group("dval"))
                dcomp = _safe_float(m_es.group("dcomp"))
                if dval is not None:
                    rec["_evorank_dval"].append(dval)
                if dcomp is not None:
                    rec["_evorank_dcomp"].append(dcomp)
                continue

    for method, rec in summary.items():
        rs = rec.get("rank_start")
        re_ = rec.get("rank_end")
        if rs is not None and re_ is not None:
            rec["rank_delta"] = float(re_) - float(rs)
        val_hist = [float(v) for v in rec.get("_val_hist", [])]
        if val_hist:
            low = all(0.4729 - 1e-6 <= v <= 0.5271 + 1e-6 for v in val_hist)
            rec["stuck_majority_band"] = "yes" if low else "no"
        rec["evorank_es_events"] = len(rec.get("_evorank_dval", [])) if method == "evorank" else ""
        rec["evorank_delta_val_loss_mean"] = _mean(rec.get("_evorank_dval", [])) if method == "evorank" else ""
        rec["evorank_delta_complexity_mean"] = _mean(rec.get("_evorank_dcomp", [])) if method == "evorank" else ""
        # 清理内部字段
        for k in ["_val_hist", "_evorank_dval", "_evorank_dcomp"]:
            rec.pop(k, None)
    return summary


def write_csv(summary: Dict[str, Dict[str, object]], out_csv: Path) -> None:
    fields = [
        "method",
        "metric_key",
        "best_val",
        "final_val",
        "rank_start",
        "rank_end",
        "rank_delta",
        "num_eval_points",
        "stuck_majority_band",
        "evorank_es_events",
        "evorank_delta_val_loss_mean",
        "evorank_delta_complexity_mean",
    ]
    order = ["lora", "adalora", "evorank", "sora"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method in order:
            if method in summary:
                writer.writerow(summary[method])
        for method in summary:
            if method not in order:
                writer.writerow(summary[method])


def write_md(
    main_summary: Dict[str, Dict[str, object]],
    appendix_summary: Dict[str, Dict[str, object]],
    main_log: Path,
    appendix_log: Path,
    out_md: Path,
) -> None:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("# RTE Main vs Appendix Protocol")
    lines.append("")
    lines.append(f"- Generated at: {now}")
    lines.append(f"- Main log: `{main_log}`")
    lines.append(f"- Appendix log: `{appendix_log}`")
    lines.append("")
    lines.append("## Main Table Protocol")
    lines.append("- Single-stage training, unified budget/hyperparameters.")
    lines.append("- SoRA uses `no_schedule` by default in main table.")
    lines.append("- EvoRank keeps `lambda_c=0.001` in fair script; lambda sweep runs separately for diagnosis.")
    lines.append("")
    lines.append("## Appendix Protocol")
    lines.append("- `schedule_dense` path is isolated as appendix only.")
    lines.append("- Not mixed into main-table fairness comparison.")
    lines.append("")
    lines.append("## Snapshot (from available logs)")
    lines.append("| Method | Main best | Main final | Main rank(start->end) | Stuck in 0.4729~0.5271 |")
    lines.append("|---|---:|---:|---|---|")
    for method in ["lora", "adalora", "evorank", "sora"]:
        rec = main_summary.get(method, {})
        best_v = rec.get("best_val")
        final_v = rec.get("final_val")
        rs = rec.get("rank_start")
        re_ = rec.get("rank_end")
        stuck = rec.get("stuck_majority_band", "")
        best_s = f"{float(best_v):.4f}" if isinstance(best_v, (float, int)) else "-"
        final_s = f"{float(final_v):.4f}" if isinstance(final_v, (float, int)) else "-"
        rank_s = "-"
        if isinstance(rs, (float, int)) and isinstance(re_, (float, int)):
            rank_s = f"{float(rs):.2f}->{float(re_):.2f}"
        lines.append(f"| {method} | {best_s} | {final_s} | {rank_s} | {stuck or '-'} |")
    if "sora" in appendix_summary:
        ap = appendix_summary["sora"]
        lines.append("")
        lines.append("## SoRA Appendix Comparison")
        main_best = main_summary.get("sora", {}).get("best_val")
        ap_best = ap.get("best_val")
        if isinstance(main_best, (float, int)) and isinstance(ap_best, (float, int)):
            lines.append(
                f"- SoRA best (main no-schedule): **{float(main_best):.4f}**; "
                f"SoRA best (appendix schedule-dense): **{float(ap_best):.4f}**."
            )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RTE diagnosis logs.")
    parser.add_argument("--main_log", type=str, default="logs/fair_glue_deberta_rte_ddp.out")
    parser.add_argument(
        "--appendix_log",
        type=str,
        default="logs/fair_glue_deberta_rte_sora_schedule_dense_appendix.out",
    )
    parser.add_argument("--out_csv", type=str, default="rte_diagnosis.csv")
    parser.add_argument("--out_md", type=str, default="rte_main_vs_appendix.md")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    main_log = (root / args.main_log).resolve()
    appendix_log = (root / args.appendix_log).resolve()
    out_csv = (root / args.out_csv).resolve()
    out_md = (root / args.out_md).resolve()

    main_summary = parse_rte_log(main_log)
    appendix_summary = parse_rte_log(appendix_log)
    write_csv(main_summary, out_csv)
    write_md(main_summary, appendix_summary, main_log, appendix_log, out_md)
    print(f"Wrote diagnosis CSV to {out_csv}")
    print(f"Wrote protocol note to {out_md}")


if __name__ == "__main__":
    main()
