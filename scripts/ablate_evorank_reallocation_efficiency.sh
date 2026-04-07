#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs runs artifacts

# Reallocation 组合爆炸防护消融
# - capped_k8: 论文/实现默认 top-k cross 限流
# - unlimited: 关闭 reallocation 候选上限，观察 wall-clock 与显存/峰值耗时退化
#
# 说明:
# - 该脚本主要看效率，不与主精度消融混在一起。
# - 选择 SST-2 做代表性任务；使用多 seed 估计均值/方差，避免单次偶然波动。

COMMON_ARGS=(
  --methods evorank
  --task_name sst2
  --model_name microsoft/deberta-v3-base
  --target_rank 8
  --evorank_r_max 16
  --lora_alpha 16
  --max_train_steps 600
  --batch_size 8
  --max_length 128
  --lr 8e-4
  --warmup_ratio 0.06
  --weight_decay 0.1
  --max_grad_norm 0.1
  --T_es 100
  --mini_val_k 8
  --lambda_c 0.001
  --complexity_mode rank_sum
  --lambda_pop 16
  --population_strategy all
  --expand_init_mode gradient
  --evo_rho 0.9
  --evo_p_g 0.8
  --evo_p_p 0.1
  --evo_H_g 2
  --evo_H_p 3
  --evo_cooldown_steps 2
  --evo_allow_reallocation
  --evo_include_noop_candidate
  --seed_list 0 42 100
  --verify_n_samples 0
  --output_dir artifacts
)

run_case() {
  local case_name="$1"
  shift
  echo "=== [$(date '+%F %T')] Start ${case_name} ==="
  torchrun --nproc_per_node=2 --master_port="$1" run_benchmark.py \
    --ddp \
    "${COMMON_ARGS[@]}" \
    --log_dir "runs/evorank_reallocation_efficiency/${case_name}" \
    --export_csv "results_evorank_reallocation_${case_name}.csv" \
    "${@:2}"
  echo "=== [$(date '+%F %T')] Done ${case_name} ==="
}

run_case capped_k8 29620 --evo_max_reallocate_candidates 8
run_case unlimited 29621 --evo_max_reallocate_candidates 0

echo "Reallocation efficiency ablations completed."
