#!/bin/bash
# ============================================================================
# 公平对比: GLUE × DeBERTa-v3-base × 全方法（逐任务超参对齐 AdaLoRA 官方）
# ============================================================================
#
# 超参来源：AdaLoRA 官方仓库 (ICLR 2023) DeBERTa-v3-base 在 8 个 GLUE 子任务上
# 的独立超参（NLU/scripts/run_debertav3_*.sh），是唯一在 DeBERTa-v3-base 上
# 覆盖全部 GLUE 子任务的基准。
#
# AdaLoRA 官方参数映射：
#   init_warmup → --adalora_tinit    (初始满秩训练步数)
#   final_warmup → --adalora_tfinal  (末尾固定秩步数)
#   mask_interval → --adalora_delta_t (秩调整间隔步数)
#   reg_orth_coef → --adalora_orth_reg_weight
#
# 公平原则：
#   - 每个任务的 lr / epochs / max_seq_length / lora_alpha 取自 AdaLoRA 脚本
#   - 所有方法共享同一套 target_modules（6 类，含 attention.output.dense）
#   - 所有方法共享同一套 dropout（protocol_dropout=0.05）
#   - weight_decay 统一 0.01（AdaLoRA 各任务 0~0.1，取折中；stsb 保留 0.1）
#   - batch_size 统一 32（与 AdaLoRA 官方一致）
#   - warmup_ratio 统一 0.06（SoRA/AdaLoRA 均用此值）
#   - seed_list 沿用 SoRA 论文的 5 种子：0 21 42 81 100
#   - SoRA 特有参数按论文 no-schedule 主线
#   - 8 个 GLUE 任务分批并行（PARALLEL_JOBS 控制每批同时跑几个任务，默认 2）。
#     每个任务内部 nproc_per_node=2（双卡 DDP），多个任务共享同一对 GPU。
#     DeBERTa-v3-base 单任务约 4-8GB/卡，A800 80GB 可轻松并行 2-4 个任务。
# ============================================================================

set -euo pipefail
mkdir -p logs runs artifacts

METHODS="lora adalora evorank sora toplora flatlora pissa"
MODEL="microsoft/deberta-v3-base"
SEEDS="0 21 42 81 100"
PROTOCOL="controlled_fair"
PROTOCOL_DROPOUT=0.05

# 每批同时跑几个任务（每个任务内部仍为 2 卡 DDP），按显存余量调整
PARALLEL_JOBS="${PARALLEL_JOBS:-2}"
# 并行时每个 torchrun 必须使用不同 master_port
BASE_MASTER_PORT="${BASE_MASTER_PORT:-29500}"
MASTER_PORT_STEP="${MASTER_PORT_STEP:-100}"

# ---- 启动单个任务（第一个参数为该作业的 master_port）----
run_task() {
  local MASTER_PORT=$1
  local TASK=$2
  local LR=$3
  local EPOCHS=$4
  local MAX_LEN=$5
  local ALPHA=$6
  local WD=$7
  local TINIT=$8          # adalora_tinit  (= AdaLoRA init_warmup)
  local TFINAL=$9         # adalora_tfinal (= AdaLoRA final_warmup)
  local DELTA_T=${10}     # adalora_delta_t (= AdaLoRA mask_interval)
  local ORTH_REG=${11}    # adalora_orth_reg_weight (= AdaLoRA reg_orth_coef)

  echo "================================================================"
  echo " Task: $TASK | master_port=$MASTER_PORT | lr=$LR epochs=$EPOCHS maxlen=$MAX_LEN alpha=$ALPHA wd=$WD"
  echo " AdaLoRA: tinit=$TINIT tfinal=$TFINAL deltaT=$DELTA_T orth=$ORTH_REG"
  echo "================================================================"

  nohup torchrun --nproc_per_node=2 --master_port="$MASTER_PORT" \
    run_benchmark.py \
    --ddp \
    --methods $METHODS \
    --comparison_protocol $PROTOCOL \
    --protocol_dropout $PROTOCOL_DROPOUT \
    --module_preset default \
    --flatlora_rho 0.05 \
    --task_list $TASK \
    --model_list $MODEL \
    --target_rank 8 \
    --lora_alpha $ALPHA \
    --epochs $EPOCHS \
    --batch_size 32 \
    --max_length $MAX_LEN \
    --lr $LR \
    --warmup_ratio 0.06 \
    --weight_decay $WD \
    --max_grad_norm 1.0 \
    --adalora_tinit $TINIT \
    --adalora_tfinal $TFINAL \
    --adalora_delta_t $DELTA_T \
    --adalora_orth_reg_weight $ORTH_REG \
    --sora_sparse_lambda 10 \
    --sora_sparse_lambda_2 3e-4 \
    --lambda_c 0.0 \
    --expand_init_mode gradient \
    --mini_val_k 8 \
    --evo_alpha_u 1.0 \
    --evo_p_p 0.05 \
    --evo_H_p 4 \
    --evo_max_reallocate_candidates 16 \
    --verify_n_samples 0 \
    --seed_list $SEEDS \
    --log_dir runs/fair_glue_deberta_${TASK} \
    --output_dir artifacts \
    --export_csv results_fair_glue_deberta_${TASK}.csv \
    > logs/fair_glue_deberta_${TASK}.out 2>&1
}

# ============================================================================
#  逐任务参数（来自 AdaLoRA NLU/scripts/run_debertav3_*.sh）
# ============================================================================
#
# AdaLoRA 官方各任务参数表：
# ┌────────┬────────┬────────┬────────┬───────┬──────┬──────────┬────────────┬──────────────┬──────────┐
# │ task   │ lr     │ epochs │ maxlen │ alpha │  wd  │ tinit    │ tfinal     │ delta_t      │ orth_reg │
# ├────────┼────────┼────────┼────────┼───────┼──────┼──────────┼────────────┼──────────────┼──────────┤
# │ cola   │ 8e-4   │ 25     │ 64     │ 32    │ 0    │ 800      │ 3500       │ 10           │ 0.1      │
# │ mnli   │ 5e-4   │ 7      │ 256    │ 16    │ 0    │ 8000     │ 50000      │ 100          │ 0.1      │
# │ mrpc   │ 1e-3   │ 30     │ 320    │ 32    │ 0.01 │ 600      │ 1800       │ 1            │ 0.1      │
# │ qqp    │ 8e-4   │ 5      │ 320    │ 16    │ 0.01 │ 8000     │ 25000      │ 100          │ 0.1      │
# │ qnli   │ 5e-4   │ 5      │ 512    │ 32    │ 0.01 │ 2000     │ 8000       │ 100          │ 0.1      │
# │ rte    │ 1.2e-3 │ 50     │ 320    │ 32    │ 0.01 │ 600      │ 1800       │ 1            │ 0.3      │
# │ sst2   │ 8e-4   │ 24     │ 128    │ 16    │ 0.01 │ 6000     │ 22000      │ 100          │ 0.1      │
# │ stsb   │ 2.2e-3 │ 25     │ 128    │ 32    │ 0.1  │ 800      │ 2000       │ 10           │ 0.3      │
# └────────┴────────┴────────┴────────┴───────┴──────┴──────────┴────────────┴──────────────┴──────────┘

# --- 8 个任务分批并行（PARALLEL_JOBS 个一批，共享双卡）---
# 任务列表：port_offset  task   lr      epochs maxlen alpha wd    tinit tfinal delta_t orth
TASKS=(
  "0  cola   8e-4    25   64   32  0.01   800   3500   10   0.1"
  "1  mnli   5e-4    7    256  16  0.01   8000  50000  100  0.1"
  "2  mrpc   1e-3    30   320  32  0.01   600   1800   1    0.1"
  "3  qqp    8e-4    5    320  16  0.01   8000  25000  100  0.1"
  "4  qnli   5e-4    5    512  32  0.01   2000  8000   100  0.1"
  "5  rte    1.2e-3  50   320  32  0.01   600   1800   1    0.3"
  "6  sst2   8e-4    24   128  16  0.01   6000  22000  100  0.1"
  "7  stsb   2.2e-3  25   128  32  0.1    800   2000   10   0.3"
)

TOTAL=${#TASKS[@]}
FAIL=0

for ((START=0; START<TOTAL; START+=PARALLEL_JOBS)); do
  PIDS=()
  BATCH_NAMES=()
  for ((J=START; J<START+PARALLEL_JOBS && J<TOTAL; J++)); do
    read -r IDX TASK LR EPOCHS MAX_LEN ALPHA WD TINIT TFINAL DELTA_T ORTH_REG <<< "${TASKS[$J]}"
    PORT=$((BASE_MASTER_PORT + IDX * MASTER_PORT_STEP))
    BATCH_NAMES+=("$TASK")
    run_task "$PORT" "$TASK" "$LR" "$EPOCHS" "$MAX_LEN" "$ALPHA" "$WD" "$TINIT" "$TFINAL" "$DELTA_T" "$ORTH_REG" &
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
