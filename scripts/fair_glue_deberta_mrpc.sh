#!/bin/bash
# MRPC：lr=1e-3 epochs=30 maxlen=320 alpha=32 wd=0.01 | AdaLoRA 600/1800/1/0.1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/fair_glue_deberta_common.sh
source "${SCRIPT_DIR}/fair_glue_deberta_common.sh"
mkdir -p logs runs artifacts
run_task mrpc 1e-3 30 320 32 0.01 600 1800 1 0.1
