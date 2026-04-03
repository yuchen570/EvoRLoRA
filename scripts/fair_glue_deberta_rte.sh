#!/bin/bash
# ============================================================================
# 公平对比: RTE 单任务 × DeBERTa-v3-base × 全方法
# ----------------------------------------------------------------------------
# RTE 是极小数据集 (2.5k 训练样本)。
# 实测在统一大 lr (1.2e-3) 下 LoRA 会早期出现 NaN，导致验证恒定预测。
# 这里改为跨方法统一且稳定的 lr=2e-4 做公平横向对比。
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
  --target_modules query_proj,key_proj,value_proj,intermediate.dense,output.dense \
  --epochs 50 \
  --batch_size 32 \
  --max_length 320 \
  --lr 2e-4 \
  --warmup_ratio 0.06 \
  --weight_decay 0.1 \
  --max_grad_norm 0.1 \
  --adalora_delta_t 100 \
  --adalora_orth_reg_weight 0.1 \
  --lora_ga_batches 8 \
  --lora_ga_stable_gamma 16 \
  --sora_sparse_lambda 10 \
  --sora_sparse_lambda_2 1e-4 \
  --expand_init_mode gradient \
  --seed_list 0 21 42 81 100 \
  --log_dir runs/fair_glue_deberta_rte_ddp \
  --output_dir artifacts \
  --export_csv results_fair_glue_deberta_rte_ddp.csv \
  > logs/fair_glue_deberta_rte_ddp.out 2>&1 &

echo "Started fair RTE comparison. Check logs/fair_glue_deberta_rte_ddp.out"
