#!/usr/bin/env python3
from __future__ import annotations
import csv
import pathlib
import re
from typing import Dict


LOG_DIR = pathlib.Path("logs")
OUT_CSV = pathlib.Path("results_rte_method_faithful_summary.csv")

PAT_EVAL = re.compile(
    r"\[(?P<method>[a-zA-Z0-9\-]+)\] epoch=(?P<ep>\d+)/(?P<epn>\d+) step=(?P<step>\d+) "
    r"val_accuracy=(?P<val>[0-9.]+) best=(?P<best>[0-9.]+)"
)
PAT_RANK = re.compile(r"avg_rank=(?P<avg>[0-9.]+)\s+total_active=(?P<active>\d+)/(?P<cap>\d+)")


def parse_log(path: pathlib.Path) -> Dict[str, str]:
    best_acc = None
    last_acc = None
    last_epoch = None
    method = ""
    last_avg_rank = ""
    last_active = ""
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = PAT_EVAL.search(line)
        if m:
            method = m.group("method")
            best_val = float(m.group("best"))
            val = float(m.group("val"))
            ep = int(m.group("ep"))
            best_acc = best_val if best_acc is None else max(best_acc, best_val)
            last_acc = val
            last_epoch = ep
        r = PAT_RANK.search(line)
        if r:
            last_avg_rank = r.group("avg")
            last_active = f"{r.group('active')}/{r.group('cap')}"
    return {
        "log_file": str(path),
        "method": method,
        "best_accuracy": "" if best_acc is None else f"{best_acc:.4f}",
        "last_accuracy": "" if last_acc is None else f"{last_acc:.4f}",
        "last_epoch": "" if last_epoch is None else str(last_epoch),
        "last_avg_rank": last_avg_rank,
        "last_active_rank": last_active,
    }


def main() -> None:
    candidates = sorted(LOG_DIR.glob("rte_fair_*.out")) + sorted(LOG_DIR.glob("tune_evorank_rte_*.out"))
    rows = [parse_log(p) for p in candidates if p.is_file()]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "log_file",
                "method",
                "best_accuracy",
                "last_accuracy",
                "last_epoch",
                "last_avg_rank",
                "last_active_rank",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
