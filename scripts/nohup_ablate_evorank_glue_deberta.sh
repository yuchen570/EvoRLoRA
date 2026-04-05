#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs
nohup bash scripts/ablate_evorank_glue_deberta.sh > logs/ablate_evorank_glue_deberta.out 2>&1 &
echo "Started EvoRank main ablations. Check logs/ablate_evorank_glue_deberta.out"
