#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs
nohup bash scripts/ablate_evorank_reallocation_efficiency.sh > logs/ablate_evorank_reallocation_efficiency.out 2>&1 &
echo "Started EvoRank reallocation efficiency ablation. Check logs/ablate_evorank_reallocation_efficiency.out"
