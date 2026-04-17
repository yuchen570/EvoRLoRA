#!/bin/bash
# QNLI：lr=5e-4 epochs=5 maxlen=512 alpha=32 wd=0.01 | AdaLoRA 2000/8000/100/0.1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/fair_glue_deberta_common.sh
source "${SCRIPT_DIR}/fair_glue_deberta_common.sh"
mkdir -p logs runs artifacts
run_task qnli 5e-4 5 512 32 0.01 2000 8000 100 0.1
