# =============================================================================
# 公平对比 GLUE × DeBERTa-v3-base：共享变量与 run_task（由各任务脚本 source）
# =============================================================================
#
# 超参与公平原则说明见各 fair_glue_deberta_<task>.sh 头部注释；此处仅保留实现。
# 不要直接执行本文件：请运行 fair_glue_deberta_<task>.sh 或 fair_glue_deberta.sh。
# =============================================================================

METHODS="lora adalora evorank sora toplora flatlora pissa"
MODEL="microsoft/deberta-v3-base"
SEEDS="0 21 42 81 100"
PROTOCOL="controlled_fair"
PROTOCOL_DROPOUT=0.05

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
