#!/bin/bash
# ============================================================================
# RTE 方法忠实公平对比（DeBERTa-v3-base）
# ----------------------------------------------------------------------------
# 目标:
#   - 共同训练预算（epoch/batch/seq_len/seed 一致）
#   - 每个方法仅开放其“官方关键旋钮”
#   - 参数口径参考 SoRA（no-schedule）
#
# 运行方式:
#   RUN_MODE=sequential bash scripts/fair_glue_deberta_rte_method_faithful.sh
#   RUN_MODE=parallel   bash scripts/fair_glue_deberta_rte_method_faithful.sh
# ============================================================================
set -euo pipefail

RUN_MODE="${RUN_MODE:-parallel}"
mkdir -p logs runs artifacts

run_case () {
  local method="$1"
  local tag="$2"
  local port="$3"
  shift 3

  local cmd=(
    torchrun --nproc_per_node=2 --master_port="${port}"
    run_benchmark.py
    --ddp
    --methods "${method}"
    --task_name rte
    --model_name microsoft/deberta-v3-base
    --target_rank 8
    --lora_alpha 16
    --epochs 20
    --batch_size 8
    --max_length 128
    --warmup_ratio 0.06
    --weight_decay 0.1
    --max_grad_norm 0.1
    --seed 48
    --verify_n_samples 0
    --log_dir "runs/${tag}"
    --output_dir artifacts
    --export_csv "results_${tag}.csv"
  )
  cmd+=("$@")

  if [[ "${RUN_MODE}" == "parallel" ]]; then
    nohup "${cmd[@]}" > "logs/${tag}.out" 2>&1 &
    echo "Started ${method} -> logs/${tag}.out"
  else
    "${cmd[@]}" > "logs/${tag}.out" 2>&1
    echo "Finished ${method} -> logs/${tag}.out"
  fi
}

# 1) LoRA baseline（SoRA口径主超参）
run_case lora rte_fair_lora 29610 \
  --lr 8e-4

# 2) AdaLoRA（保留其官方关键项）
run_case adalora rte_fair_adalora 29611 \
  --lr 8e-4 \
  --adalora_delta_t 100 \
  --adalora_orth_reg_weight 0.1

# 3) EvoRank（按当前实现稳定口径，与 fair_glue_deberta_rte.sh 对齐）
run_case evorank rte_fair_evorank 29612 \
  --lr 8e-4 \
  --lambda_c 0.0 \
  --expand_init_mode gradient \
  --mini_val_k 8 \
  --evo_alpha_u 1.0 \
  --evo_p_p 0.05 \
  --evo_H_p 4 \
  --evo_max_reallocate_candidates 16

# 4) SoRA（no-schedule 口径）
run_case sora rte_fair_sora 29614 \
  --lr 8e-4 \
  --sora_sparse_lambda 10 \
  --sora_sparse_lambda_2 3e-4

if [[ "${RUN_MODE}" == "parallel" ]]; then
  echo "All jobs launched in parallel."
else
  echo "All jobs finished in sequential mode."
fi

