#!/bin/bash
# MNLI：lr=5e-4 epochs=7 maxlen=256 alpha=16 wd=0.01 | AdaLoRA 8000/50000/100/0.1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/fair_glue_deberta_common.sh
source "${SCRIPT_DIR}/fair_glue_deberta_common.sh"
mkdir -p logs runs artifacts
MASTER_PORT="${MASTER_PORT:-$(( ${BASE_MASTER_PORT:-29500} + 1 * ${MASTER_PORT_STEP:-100} ))}"
run_task "$MASTER_PORT" mnli 5e-4 7 256 16 0.01 8000 50000 100 0.1
