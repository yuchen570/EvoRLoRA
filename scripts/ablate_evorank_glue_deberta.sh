#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs runs artifacts

# EvoRank 主消融设计
# 1. full: 全量方法
# 2. no_complexity: 去掉 reward 中的复杂度正则
# 3. zero_init: 扩张从 cold-start 零初始化开始
# 4. no_ema: 关闭 EMA 平滑
# 5. no_persist_only: 仅关闭持久计数器门槛
# 6. no_cooldown_only: 仅关闭 cooldown
# 7. no_persist_cooldown: 同时关闭持久计数器与 cooldown（历史组合）
# 8. no_reallocation: 去掉跨层重分配
# 9. no_noop: 去掉 validation-side no-op safeguard
# 10. es_budget_light: 搜索预算偏轻（更稀疏触发、更小候选）
# 11. es_budget_heavy: 搜索预算偏重（更频繁触发、更大候选）
#
# 说明:
# - 主脚本使用 DeBERTa-v3-base + GLUE 主线任务，与当前公平脚本保持同一训练协议。
# - RTE 不放进这个主脚本，因为它在本仓库里走单独的 schedule-dense 特殊协议，混进来会引入额外混杂因素。
# - “动态阈值”家族拆成两类消融：no_ema 与 no_persist_cooldown。

COMMON_ARGS=(
  --methods evorank
  --task_list mnli sst2 cola qqp qnli mrpc stsb
  --model_list microsoft/deberta-v3-base
  --target_rank 8
  --evorank_r_max 16
  --lora_alpha 16
  --epochs 20
  --batch_size 8
  --max_length 128
  --lr 8e-4
  --warmup_ratio 0.06
  --weight_decay 0.1
  --max_grad_norm 0.1
  --T_es 200
  --mini_val_k 8
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
  --evo_max_reallocate_candidates 8
  --evo_include_noop_candidate
  --seed_list 0 21 42 81 100
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
    --log_dir "runs/evorank_ablation/${case_name}" \
    --export_csv "results_evorank_${case_name}.csv" \
    "${@:2}"
  echo "=== [$(date '+%F %T')] Done ${case_name} ==="
}

run_case full 29600 --lambda_c 0.001
run_case no_complexity 29601
run_case zero_init 29602 --expand_init_mode zero
run_case no_ema 29603 --evo_rho 0.0
run_case no_persist_only 29604 --evo_H_g 1 --evo_H_p 1
run_case no_cooldown_only 29605 --evo_cooldown_steps 0
run_case no_persist_cooldown 29606 --evo_H_g 1 --evo_H_p 1 --evo_cooldown_steps 0
run_case no_reallocation 29607 --no_evo_allow_reallocation
run_case no_noop 29608 --no_evo_include_noop_candidate
run_case es_budget_light 29609 --T_es 300 --lambda_pop 8 --mini_val_k 4
run_case es_budget_heavy 29610 --T_es 100 --lambda_pop 32 --mini_val_k 16

echo "All EvoRank main ablations completed."
