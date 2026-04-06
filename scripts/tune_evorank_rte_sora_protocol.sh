#!/bin/bash
# ============================================================================
# EvoRank RTE 调参（SoRA 协议基线）
# ----------------------------------------------------------------------------
# 关注三点:
#   1) lambda_c 强度
#   2) reallocation 组合上限
#   3) 在“动量清理已开启”前提下，阈值侧避免过度 prune
#
# 说明:
#   - 预算口径固定为 SoRA 参考（20 epoch, bsz=8, len=128）
#   - 默认串行，避免多作业互相干扰；可 RUN_MODE=parallel 切并行；可 RUN_MODE=sequential 切串行
# ============================================================================
set -euo pipefail

RUN_MODE="${RUN_MODE:-parallel}"
mkdir -p logs runs artifacts

run_case () {
  local tag="$1"
  shift

  local cmd=(
    torchrun --nproc_per_node=2 --master_port="$((29700 + (RANDOM % 200)))"
    run_benchmark.py
    --ddp
    --methods evorank
    --task_name rte
    --model_name microsoft/deberta-v3-base
    --target_rank 8
    --lora_alpha 16
    --epochs 20
    --batch_size 8
    --max_length 128
    --lr 8e-4
    --warmup_ratio 0.06
    --weight_decay 0.1
    --max_grad_norm 0.1
    --seed 48
    --verify_n_samples 0
    --expand_init_mode gradient
    --log_dir "runs/${tag}"
    --output_dir artifacts
    --export_csv "results_${tag}.csv"
  )
  cmd+=("$@")

  if [[ "${RUN_MODE}" == "parallel" ]]; then
    nohup "${cmd[@]}" > "logs/${tag}.out" 2>&1 &
    echo "Started ${tag}"
  else
    "${cmd[@]}" > "logs/${tag}.out" 2>&1
    echo "Finished ${tag}"
  fi
}

# Phase A: 主因定位（lambda_c × reallocate cap）
for lam in 0.0 1e-5 1e-4; do
  for cap in 8 16; do
    tag="tune_evorank_rte_lam${lam}_realloc${cap}"
    run_case "${tag}" \
      --lambda_c "${lam}" \
      --mini_val_k 16 \
      --evo_alpha_u 2.0 \
      --evo_p_p 0.05 \
      --evo_H_p 4 \
      --evo_max_reallocate_candidates "${cap}"
  done
done

# Phase B: 阈值/稳定性细调（固定低惩罚）
for alpha_u in 1.0 2.0; do
  for mk in 8 16; do
    tag="tune_evorank_rte_alpha${alpha_u}_mk${mk}"
    run_case "${tag}" \
      --lambda_c 0.0 \
      --mini_val_k "${mk}" \
      --evo_alpha_u "${alpha_u}" \
      --evo_p_p 0.05 \
      --evo_H_p 4 \
      --evo_max_reallocate_candidates 16
  done
done

if [[ "${RUN_MODE}" == "parallel" ]]; then
  echo "All EvoRank tuning jobs launched."
else
  echo "All EvoRank tuning jobs finished."
fi

