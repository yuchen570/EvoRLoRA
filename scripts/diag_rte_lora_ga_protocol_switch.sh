#!/bin/bash
# ============================================================================
# RTE 诊断: LoRA-GA 统一调度 vs 官方调度分支
# ----------------------------------------------------------------------------
# 统一调度: 使用公平主表调度
# 官方分支: warmup_ratio=0.03 + cosine (仅用于定位，不进主表)
# ============================================================================
set -euo pipefail

mkdir -p logs runs artifacts

nohup torchrun --nproc_per_node=2 --master_port=29531 \
  run_benchmark.py \
  --ddp \
  --methods lora-ga \
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
  --lora_ga_batches 8 \
  --seed 48 \
  --log_dir runs/rte_lora_ga_uniform_schedule \
  --output_dir artifacts \
  --export_csv results_rte_lora_ga_uniform_schedule.csv \
  > logs/rte_lora_ga_uniform_schedule.out 2>&1 &

nohup torchrun --nproc_per_node=2 --master_port=29532 \
  run_benchmark.py \
  --ddp \
  --methods lora-ga \
  --task_name rte \
  --model_name microsoft/deberta-v3-base \
  --target_rank 8 \
  --lora_alpha 16 \
  --epochs 50 \
  --batch_size 32 \
  --max_length 320 \
  --lr 1.2e-3 \
  --weight_decay 0.1 \
  --max_grad_norm 0.1 \
  --lora_ga_batches 8 \
  --lora_ga_use_official_scheduler \
  --lora_ga_official_warmup_ratio 0.03 \
  --lora_ga_official_scheduler_type cosine \
  --seed 48 \
  --log_dir runs/rte_lora_ga_official_schedule \
  --output_dir artifacts \
  --export_csv results_rte_lora_ga_official_schedule.csv \
  > logs/rte_lora_ga_official_schedule.out 2>&1 &

echo "Started LoRA-GA protocol switch diagnostics."
