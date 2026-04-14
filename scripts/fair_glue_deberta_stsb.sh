#!/bin/bash
# STS-B：lr=2.2e-3 epochs=25 maxlen=128 alpha=32 wd=0.1 | AdaLoRA 800/2000/10/0.3
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/fair_glue_deberta_common.sh
source "${SCRIPT_DIR}/fair_glue_deberta_common.sh"
mkdir -p logs runs artifacts
MASTER_PORT="${MASTER_PORT:-$(( ${BASE_MASTER_PORT:-29500} + 7 * ${MASTER_PORT_STEP:-100} ))}"
run_task "$MASTER_PORT" stsb 2.2e-3 25 128 32 0.1 800 2000 10 0.3
