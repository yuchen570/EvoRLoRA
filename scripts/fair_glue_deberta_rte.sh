#!/bin/bash
# RTE：lr=1.2e-3 epochs=50 maxlen=320 alpha=32 wd=0.01 | AdaLoRA 600/1800/1/0.3
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/fair_glue_deberta_common.sh
source "${SCRIPT_DIR}/fair_glue_deberta_common.sh"
mkdir -p logs runs artifacts
MASTER_PORT="${MASTER_PORT:-$(( ${BASE_MASTER_PORT:-29500} + 5 * ${MASTER_PORT_STEP:-100} ))}"
run_task "$MASTER_PORT" rte 1.2e-3 50 320 32 0.01 600 1800 1 0.3
