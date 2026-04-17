#!/bin/bash
# MNLI：lr=5e-4 epochs=7 maxlen=256 alpha=16 wd=0.0 | AdaLoRA 8000/50000/100/0.1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/fair_glue_deberta_common.sh
source "${SCRIPT_DIR}/fair_glue_deberta_common.sh"
mkdir -p logs runs artifacts
run_task mnli 5e-4 7 256 16 0.0 8000 50000 100 0.1
