#!/bin/bash
# ============================================================================
# 附录实验: RTE × DeBERTa-v3-base × SoRA(schedule-dense compatible path)
# ----------------------------------------------------------------------------
# 说明:
#   - 此脚本仅用于附录，不进入主表公平对比。
#   - 目的是保留与 schedule_dense 相关的可复现实验入口。
# ============================================================================
mkdir -p logs runs artifacts

nohup torchrun --nproc_per_node=2 --master_port=29511 \
  run_benchmark.py \
  --ddp \
  --methods sora \
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
  --sora_sparse_lambda 1e-3 \
  --sora_sparse_lambda_2 0 \
  --sora_lambda_schedule linear \
  --sora_max_lambda 7e-4 \
  --sora_lambda_num 7 \
  --seed 48 \
  --log_dir runs/fair_glue_deberta_rte_sora_schedule_dense_appendix \
  --output_dir artifacts \
  --export_csv results_rte_sora_schedule_dense_appendix.csv \
  > logs/fair_glue_deberta_rte_sora_schedule_dense_appendix.out 2>&1 &

echo "Started appendix SoRA schedule-dense run. Check logs/fair_glue_deberta_rte_sora_schedule_dense_appendix.out"
