#!/bin/bash
# ============================================================================
# 冒烟测试: SST-2 × DeBERTa-v3-base × 全方法 (50 步快速验证)
# 对齐 SoRA 主线超参的缩短版：
#   lr=8e-4, bsz=8, warmup_ratio=0.06, weight_decay=0.1,
#   max_grad_norm=0.1, r=8, sparse_lambda=10, sparse_lambda_2=3e-4
# 不显式传 --target_modules，让各方法走各自默认协议。
# EvoRank: --expand_init_mode gradient（仅 evorank 生效）
# ============================================================================
mkdir -p logs runs artifacts

python run_benchmark.py \
  --methods lora adalora evorank lora-ga sora \
  --task_name sst2 \
  --model_name microsoft/deberta-v3-base \
  --target_rank 8 \
  --lora_alpha 16 \
  --max_train_steps 50 \
  --batch_size 8 \
  --max_length 128 \
  --lr 8e-4 \
  --warmup_ratio 0.06 \
  --weight_decay 0.1 \
  --max_grad_norm 0.1 \
  --T_es 20 \
  --adalora_delta_t 10 \
  --adalora_orth_reg_weight 0.1 \
  --lora_ga_batches 8 \
  --sora_sparse_lambda 10 \
  --sora_sparse_lambda_2 3e-4 \
  --expand_init_mode gradient \
  --evo_max_reallocate_candidates 8 \
  --seed 42 \
  --log_dir runs/fair_smoke \
  --output_dir artifacts \
  --export_csv results_fair_smoke.csv

echo "Smoke test done. Check results_fair_smoke.csv"
