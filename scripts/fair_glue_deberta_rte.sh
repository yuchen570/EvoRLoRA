#!/bin/bash
# ============================================================================
# 公平对比: RTE 单任务 × DeBERTa-v3-base × 全方法
# ----------------------------------------------------------------------------
# 统一对齐 SoRA 官方 schedule-dense 脚本:
#   lr=1.2e-3, bsz=32, epoch=50, max_length=320,
#   sparse_lambda=1e-3, sparse_lambda_2=0,
#   lambda_schedule=linear, max_lambda=7e-4, lambda_num=7, seed=48
#
# 公平原则:
#   - 训练超参按 SoRA 官方 RTE schedule-dense 配置统一
#   - 不再显式传 --target_modules，让各方法走各自论文/官方实现的默认注入协议
# EvoRank: --expand_init_mode gradient（仅 evorank 生效）
# ============================================================================
mkdir -p logs runs artifacts

nohup torchrun --nproc_per_node=2 --master_port=29510 \
  run_benchmark.py \
  --ddp \
  --methods lora adalora evorank lora-ga sora \
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
  --adalora_delta_t 100 \
  --adalora_orth_reg_weight 0.1 \
  --lora_ga_batches 8 \
  --sora_sparse_lambda 1e-3 \
  --sora_sparse_lambda_2 0 \
  --sora_lambda_schedule linear \
  --sora_max_lambda 7e-4 \
  --sora_lambda_num 7 \
  --lambda_c 0.001 \
  --expand_init_mode gradient \
  --evo_max_reallocate_candidates 8 \
  --seed 48 \
  --log_dir runs/fair_glue_deberta_rte_ddp \
  --output_dir artifacts \
  --export_csv results_fair_glue_deberta_rte_ddp.csv \
  > logs/fair_glue_deberta_rte_ddp.out 2>&1 &

echo "Started fair RTE comparison. Check logs/fair_glue_deberta_rte_ddp.out"
