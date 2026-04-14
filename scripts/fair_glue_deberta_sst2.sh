#!/bin/bash
# SST-2：lr=8e-4 epochs=24 maxlen=128 alpha=16 wd=0.01 | AdaLoRA 6000/22000/100/0.1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/fair_glue_deberta_common.sh
source "${SCRIPT_DIR}/fair_glue_deberta_common.sh"
mkdir -p logs runs artifacts
MASTER_PORT="${MASTER_PORT:-$(( ${BASE_MASTER_PORT:-29500} + 6 * ${MASTER_PORT_STEP:-100} ))}"
run_task "$MASTER_PORT" sst2 8e-4 24 128 16 0.01 6000 22000 100 0.1
