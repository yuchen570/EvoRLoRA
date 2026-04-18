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
    --toplora_lambda_clamp 3.0 \
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
    --evo_stop_step_ratio 0.7 \
    --evo_max_reallocate_candidates 16 \
    --verify_n_samples 0 \
    --seed_list $SEEDS \
    --log_dir runs/fair_glue_deberta_${TASK} \
    --output_dir artifacts \
    --export_csv results_fair_glue_deberta_${TASK}.csv \
    > logs/fair_glue_deberta_${TASK}.out 2>&1
}
#
# 参数说明（对比算法按官方设置 + 日志诊断调整，2026-04-18）：
#   PiSSA / TopLoRA：run_benchmark.py 自动检测 lr>2e-4 时将其降至 lr/4（下限 1e-4），
#     分别对齐官方 2e-5（Llama2-7b+MetaMath）与 1e-4（Qwen2.5-3B+math_10k）scale 的
#     GLUE 小任务经验值。可通过 --pissa_lr / --toplora_lr 显式覆盖。
#   TopLoRA：新增 --toplora_lambda_clamp 3.0，将 exp(·) 输入 clamp 到 ±3，
#     缓解 seed=42 下门控指数爆炸导致的整轮崩溃（val=0）。
#   Flat-LoRA：flatlora_inject.py 已将 DDP seed 加入 local_rank 做 per-rank 独立扰动，
#     对齐官方 time-based seed 的 per-worker 扰动多样性。
#   EvoRank（基于 cola 日志 ES 事件与 val 轨迹调整）：
#     mini_val_k 8→16             减 ES 评估噪声（样本 256→512），后期 val_loss~1.5 时更鲁棒
#     evo_alpha_u 1.0→1.5         容量组合更重视梯度 g̃，抑制仅按 s̃ 剪枝
#     evo_p_g 0.8→0.75            扩张阈值放宽，平衡剪枝倾向
#     evo_p_p 0.05→0.03           剪枝更保守，减少噪声 val 触发的误剪
#     evo_H_p 4→6                 剪枝持久计数窗口延长
#     evo_cooldown_steps 2→5      减少连续反复变异
#     evo_stop_step_ratio 0.7     后 30% 训练步冻结 ES，对标 AdaLoRA tfinal
#                                 （日志证据：step>1800 后 val_loss 飙到 1.5+，ES 决策基于过拟合信号）
