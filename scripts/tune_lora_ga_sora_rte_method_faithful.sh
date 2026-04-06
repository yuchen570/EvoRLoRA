#!/bin/bash
# ============================================================================
# RTE 方法忠实调参：LoRA-GA + SoRA（仅方法内参数，不改统一预算）
# ----------------------------------------------------------------------------
# 统一预算（固定，不调）:
#   - task=rte, model=deberta-v3-base
#   - epochs=20, batch_size=8, max_length=128
#   - weight_decay=0.1, max_grad_norm=0.1, seed=48
#
# 可调范围（仅方法内）:
#   - LoRA-GA: lr / lora_ga_batches / stable_gamma / 官方调度开关
#   - SoRA: lr / sparse_lambda / sparse_lambda_2 / lambda_warmup_steps
#
# 用法:
#   RUN_MODE=sequential bash /D/EvoRLoRA/scripts/tune_lora_ga_sora_rte_method_faithful.sh
#   RUN_MODE=parallel   bash /D/EvoRLoRA/scripts/tune_lora_ga_sora_rte_method_faithful.sh
# ============================================================================
set -euo pipefail

RUN_MODE="${RUN_MODE:-sequential}"
mkdir -p logs runs artifacts

run_case () {
  local tag="$1"
  shift

  local cmd=(
    torchrun --nproc_per_node=2 --master_port="$((29800 + (RANDOM % 120)))"
    run_benchmark.py
    --ddp
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
    echo "Started ${tag}"
  else
    "${cmd[@]}" > "logs/${tag}.out" 2>&1
    echo "Finished ${tag}"
  fi
}

# ----------------------------------------------------------------------------
# LoRA-GA: 方法忠实 + 防塌缩扫描
# ----------------------------------------------------------------------------
for lr in 5e-5 8e-5 1e-4; do
  for batches in 4 8; do
    for gamma in 12 16; do
      tag="tune_lora_ga_rte_lr${lr}_b${batches}_g${gamma}"
      run_case "${tag}" \
        --methods lora-ga \
        --lr "${lr}" \
        --lora_ga_batches "${batches}" \
        --lora_ga_stable_gamma "${gamma}" \
        --lora_ga_use_rslora \
        --lora_ga_use_official_scheduler \
        --lora_ga_official_warmup_ratio 0.03 \
        --lora_ga_official_scheduler_type cosine
    done
  done
done

# ----------------------------------------------------------------------------
# SoRA: 方法忠实 no-schedule 主线 + 稀疏强度扫描（避免过快塌缩）
# ----------------------------------------------------------------------------
for lr in 6e-4 8e-4; do
  for lam in 0.5 1 2; do
    for lam2 in 1e-4 2e-4 3e-4; do
      for warm in 0 200 400; do
        tag="tune_sora_rte_lr${lr}_lam${lam}_lam2_${lam2}_w${warm}"
        run_case "${tag}" \
          --methods sora \
          --lr "${lr}" \
          --sora_sparse_lambda "${lam}" \
          --sora_sparse_lambda_2 "${lam2}" \
          --sora_lambda_warmup_steps "${warm}"
      done
    done
  done
done

if [[ "${RUN_MODE}" == "parallel" ]]; then
  echo "All LoRA-GA/SoRA tuning jobs launched."
else
  echo "All LoRA-GA/SoRA tuning jobs finished."
fi

