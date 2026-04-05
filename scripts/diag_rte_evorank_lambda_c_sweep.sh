#!/bin/bash
# ============================================================================
# RTE 诊断: EvoRank lambda_c 三档对照 (0 / 1e-4 / 1e-3)
# ----------------------------------------------------------------------------
# 目的:
#   - 定位复杂度正则是否在小样本任务上主导了 ES 决策
#   - 对照每档 lambda_c 的 best acc、rank 轨迹与 ES delta 指标
# ============================================================================
set -euo pipefail

mkdir -p logs runs artifacts

run_case () {
  local lam="$1"
  local tag="$2"
  local port="$3"
  nohup torchrun --nproc_per_node=2 --master_port="${port}" \
    run_benchmark.py \
    --ddp \
    --methods evorank \
    --task_name rte \
    --model_name microsoft/deberta-v3-base \
    --target_rank 8 \
    --lora_alpha 16 \
    --epochs 50 \
    --batch_size 32 \
    --max_length 320 \
    --lr 1.2e-3 \
    --warmup_ratio 0.06 \
    --weight_decay 0.1 \
    --max_grad_norm 0.1 \
    --lambda_c "${lam}" \
    --expand_init_mode gradient \
    --evo_max_reallocate_candidates 8 \
    --seed 48 \
    --log_dir "runs/rte_evorank_lambda_sweep_${tag}" \
    --output_dir artifacts \
    --export_csv "results_rte_evorank_lambda_${tag}.csv" \
    > "logs/rte_evorank_lambda_${tag}.out" 2>&1 &
  echo "Started EvoRank lambda_c=${lam} run -> logs/rte_evorank_lambda_${tag}.out"
}

run_case "0.0" "0" "29521"
run_case "1e-4" "1e-4" "29522"
run_case "1e-3" "1e-3" "29523"
