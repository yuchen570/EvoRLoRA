#!/bin/bash
# ============================================================================
# 公平对比: RTE 单任务 × DeBERTa-v3-base × 全方法
# ----------------------------------------------------------------------------
# RTE 是极小数据集 (2.5k 训练样本), 需要更多训练轮数与更大学习率.
# 参考:
#   - SoRA (run_glue_sora_schedule_dense.sh): lr=1.2e-3, epochs=50, bsz=32
#   - AdaLoRA (run_debertav3_rte.sh): lr=1.2e-3, epochs=50, bsz=32
# 两者在 RTE 上完全一致的超参, 直接采用统一配置.
# ============================================================================
mkdir -p logs runs artifacts

nohup torchrun --nproc_per_node=2 --master_port=29500 \
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
  --lr 1.2e-3 \
  --warmup_ratio 0.06 \
  --weight_decay 0.1 \
  --max_grad_norm 0.1 \
  --adalora_delta_t 100 \
  --adalora_orth_reg_weight 0.1 \
  --lora_ga_batches 8 \
  --lora_ga_stable_gamma 16 \
  --sora_sparse_lambda 10 \
  --sora_sparse_lambda_2 3e-4 \
  --seed_list 0 21 42 81 100 \
  --log_dir runs/fair_glue_deberta_rte_ddp \
  --output_dir artifacts \
  --export_csv results_fair_glue_deberta_rte_ddp.csv \
  > logs/fair_glue_deberta_rte_ddp.out 2>&1 &

echo "Started fair RTE comparison. Check logs/fair_glue_deberta_rte_ddp.out"
