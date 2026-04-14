#!/bin/bash
# ============================================================================
# 公平对比: GLUE × DeBERTa-v3-base × 全方法（逐任务超参对齐 AdaLoRA 官方）
# ============================================================================
#
# 本文件仅负责按批并行调度；每个任务的超参与说明在对应脚本中：
#   fair_glue_deberta_<task>.sh
# 共享 torchrun 命令体在：fair_glue_deberta_common.sh
#
# 用法：
#   - 跑全部 8 任务（与旧版行为一致，PARALLEL_JOBS 控制每批并行数）：
#       bash scripts/fair_glue_deberta.sh
#   - 只跑单个任务：
#       bash scripts/fair_glue_deberta_sst2.sh
#
# AdaLoRA 字段映射：init_warmup→tinit, final_warmup→tfinal,
#   mask_interval→delta_t, reg_orth_coef→orth_reg_weight。
# 公平原则（lr/epochs/maxlen/alpha 等按任务取自 AdaLoRA NLU 脚本）见各任务脚本注释。
# ============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

PARALLEL_JOBS="${PARALLEL_JOBS:-2}"

TASK_SCRIPTS=(
  fair_glue_deberta_cola.sh
  fair_glue_deberta_mnli.sh
  fair_glue_deberta_mrpc.sh
  fair_glue_deberta_qqp.sh
  fair_glue_deberta_qnli.sh
  fair_glue_deberta_rte.sh
  fair_glue_deberta_sst2.sh
  fair_glue_deberta_stsb.sh
)

TOTAL=${#TASK_SCRIPTS[@]}
FAIL=0

for ((START=0; START<TOTAL; START+=PARALLEL_JOBS)); do
  PIDS=()
  BATCH_NAMES=()
  for ((J=START; J<START+PARALLEL_JOBS && J<TOTAL; J++)); do
    s="${TASK_SCRIPTS[$J]}"
    name="${s#fair_glue_deberta_}"
    name="${name%.sh}"
    BATCH_NAMES+=("$name")
    bash "${SCRIPT_DIR}/${s}" &
    PIDS+=($!)
  done
  echo ""
  echo ">>> Batch [$(( START/PARALLEL_JOBS + 1 ))]: ${BATCH_NAMES[*]} (PIDs: ${PIDS[*]}). Waiting..."
  for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
  done
  echo ">>> Batch [$(( START/PARALLEL_JOBS + 1 ))]: ${BATCH_NAMES[*]} done."
done

echo ""
if [ "$FAIL" -ne 0 ]; then
  echo "One or more tasks failed. Check logs/fair_glue_deberta_*.out"
  exit 1
fi
echo "All 8 GLUE tasks finished successfully. Check logs/fair_glue_deberta_*.out"
