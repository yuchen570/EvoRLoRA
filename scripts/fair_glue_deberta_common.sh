# =============================================================================
# 公平对比 GLUE × DeBERTa-v3-base：共享变量与 run_task（由各任务脚本 source）
# =============================================================================
#
# 超参与公平原则说明见各 fair_glue_deberta_<task>.sh 头部注释；此处仅保留实现。
# 不要直接执行本文件：请运行 fair_glue_deberta_<task>.sh 或 fair_glue_deberta.sh。
#
# 环境变量（可选）：
#   NPROC_PER_NODE        torchrun 进程数；未设置时按本机 GPU 数检测，且公平默认上限为 2。
#   PER_DEVICE_BATCH_SIZE 每进程 batch（传给 run_benchmark.py --batch_size）
#   ADALORA_REF_GLOBAL_BATCH  AdaLoRA NLU 参考脚本的全局 batch（默认 32 = 1×32），用于缩放 tinit/tf/delta
# =============================================================================

METHODS="lora adalora evorank sora toplora flatlora pissa"
MODEL="microsoft/deberta-v3-base"
SEEDS="0 21 42 81 100"
PROTOCOL="controlled_fair"
PROTOCOL_DROPOUT=0.05

fair_default_nproc() {
  local g
  if command -v nvidia-smi >/dev/null 2>&1; then
    g=$(nvidia-smi -L 2>/dev/null | wc -l)
    g=${g//[[:space:]]/}
    [[ -z "$g" || "$g" -lt 1 ]] && g=1
    (( g > 2 )) && g=2
    echo "$g"
  else
    echo 1
  fi
}

NPROC_PER_NODE="${NPROC_PER_NODE:-$(fair_default_nproc)}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
ADALORA_REF_GLOBAL_BATCH="${ADALORA_REF_GLOBAL_BATCH:-32}" # AdaLoRA NLU 参考：1 GPU × bs32

scale_adalora_steps() {
  local ref_steps=$1
  local actual_global_batch=$(( NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE ))
  local scaled

  # 全局 batch 变化时按「优化器步数」比例缩放 AdaLoRA 步级超参，与参考脚本在相近 epoch 覆盖下对齐。
  scaled=$(( (ref_steps * ADALORA_REF_GLOBAL_BATCH + actual_global_batch / 2) / actual_global_batch ))
  if (( scaled < 1 )); then
    scaled=1
  fi
  echo "$scaled"
}

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

  local SCALED_TINIT SCALED_TFINAL SCALED_DELTA_T
  SCALED_TINIT="$(scale_adalora_steps "$TINIT")"
  SCALED_TFINAL="$(scale_adalora_steps "$TFINAL")"
  SCALED_DELTA_T="$(scale_adalora_steps "$DELTA_T")"

  echo "================================================================"
  echo " Task: $TASK | master_port=$MASTER_PORT | lr=$LR epochs=$EPOCHS maxlen=$MAX_LEN alpha=$ALPHA wd=$WD"
  echo " Runtime: nproc=$NPROC_PER_NODE per_device_bs=$PER_DEVICE_BATCH_SIZE global_bs=$(( NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE ))"
  echo " AdaLoRA(ref@global_bs=${ADALORA_REF_GLOBAL_BATCH}): tinit=$TINIT tfinal=$TFINAL deltaT=$DELTA_T orth=$ORTH_REG"
  echo " AdaLoRA(scaled): tinit=$SCALED_TINIT tfinal=$SCALED_TFINAL deltaT=$SCALED_DELTA_T"
  echo "================================================================"

  nohup torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
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
    --batch_size "$PER_DEVICE_BATCH_SIZE" \
    --max_length $MAX_LEN \
    --lr $LR \
    --warmup_ratio 0.06 \
    --weight_decay $WD \
    --max_grad_norm 1.0 \
    --adalora_tinit "$SCALED_TINIT" \
    --adalora_tfinal "$SCALED_TFINAL" \
    --adalora_delta_t "$SCALED_DELTA_T" \
    --adalora_orth_reg_weight $ORTH_REG \
    --sora_sparse_lambda 10 \
    --sora_sparse_lambda_2 3e-4 \
    --lambda_c 0.0 \
    --expand_init_mode gradient \
    --evo_compensation_mode B \
    --mini_val_k 16 \
    --evo_alpha_u 1.5 \
    --evo_p_g 0.75 \
    --evo_p_p 0.03 \
    --evo_H_p 6 \
    --evo_cooldown_steps 5 \
    --evo_max_reallocate_candidates 16 \
    --verify_n_samples 0 \
    --seed_list $SEEDS \
    --log_dir runs/fair_glue_deberta_${TASK} \
    --output_dir artifacts \
    --export_csv results_fair_glue_deberta_${TASK}.csv \
    > logs/fair_glue_deberta_${TASK}.out 2>&1
}
