#!/bin/bash
# CoLA：lr=8e-4 epochs=25 maxlen=64 alpha=32 wd=0.0 | AdaLoRA tinit/tf/delta/orth=800/3500/10/0.1
# 公平协议与全方法列表见 fair_glue_deberta_common.sh；从仓库根目录执行 torchrun。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/fair_glue_deberta_common.sh
source "${SCRIPT_DIR}/fair_glue_deberta_common.sh"
mkdir -p logs runs artifacts
MASTER_PORT="${MASTER_PORT:-$(( ${BASE_MASTER_PORT:-29500} + 0 * ${MASTER_PORT_STEP:-100} ))}"
run_task "$MASTER_PORT" cola 8e-4 25 64 32 0.0 800 3500 10 0.1
