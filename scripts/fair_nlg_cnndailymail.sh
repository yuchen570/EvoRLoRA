#!/bin/bash
# ============================================================================
# 公平对比: CNN/DailyMail × BART-large × 全方法
# ----------------------------------------------------------------------------
# 参考:
#   - AdaLoRA (run_bart_cnndailymail.sh): lr=5e-4, epochs=15, per_device_bsz=4,
#     lora_alpha=32, target_modules=q_proj,k_proj,v_proj,out_proj,fc1,fc2,
#     max_source_length=1024, max_target_length=160, warmup=3000 steps
# EvoRank: --expand_init_mode gradient（仅 evorank 生效）
# ============================================================================
mkdir -p logs runs artifacts

nohup torchrun --nproc_per_node=2 --master_port=29520 \
  run_benchmark.py \
  --ddp \
  --task_type nlg \
  --nlg_dataset_name cnn_dailymail \
  --task_name cnn_dailymail \
  --methods lora adalora evorank lora-ga sora \
  --model_name facebook/bart-large \
  --target_rank 8 \
  --lora_alpha 32 \
  --target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
  --epochs 15 \
  --batch_size 16 \
  --max_length 1024 \
  --max_target_length 160 \
  --generation_max_new_tokens 160 \
  --lr 5e-4 \
  --warmup_ratio 0.06 \
  --weight_decay 0.01 \
  --max_grad_norm 0.1 \
  --adalora_delta_t 100 \
  --adalora_orth_reg_weight 0.1 \
  --lora_ga_batches 8 \
  --sora_sparse_lambda 1e-3 \
  --sora_sparse_lambda_2 1e-4 \
  --expand_init_mode gradient \
  --evo_max_reallocate_candidates 8 \
  --seed 42 \
  --log_dir runs/fair_nlg_cnndm_ddp \
  --output_dir artifacts \
  --export_csv results_fair_nlg_cnndm_ddp.csv \
  > logs/fair_nlg_cnndm_ddp.out 2>&1 &

echo "Started fair CNN/DailyMail comparison (BART-large, all methods). Check logs/fair_nlg_cnndm_ddp.out"
