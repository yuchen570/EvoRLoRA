#!/bin/bash
# ============================================================================
# 公平对比: XSum × BART-large × 全方法
# ----------------------------------------------------------------------------
# 参考:
#   - AdaLoRA (run_bart_xsum.sh): lr=5e-4, epochs=25, per_device_bsz=8,
#     lora_alpha=32, target_modules=q_proj,k_proj,v_proj,out_proj,fc1,fc2,
#     warmup=3000 steps (对应约 warmup_ratio≈0.06)
# ============================================================================
mkdir -p logs runs artifacts

nohup torchrun --nproc_per_node=2 --master_port=29530 \
  run_benchmark.py \
  --ddp \
  --task_type nlg \
  --nlg_dataset_name xsum \
  --task_name xsum \
  --methods lora adalora evorank lora-ga sora \
  --model_name facebook/bart-large \
  --target_rank 8 \
  --lora_alpha 32 \
  --target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
  --epochs 25 \
  --batch_size 16 \
  --max_length 768 \
  --max_target_length 64 \
  --generation_max_new_tokens 64 \
  --lr 5e-4 \
  --warmup_ratio 0.06 \
  --weight_decay 0.01 \
  --max_grad_norm 0.1 \
  --adalora_delta_t 100 \
  --adalora_orth_reg_weight 0.1 \
  --lora_ga_batches 8 \
  --lora_ga_stable_gamma 16 \
  --sora_sparse_lambda 10 \
  --sora_sparse_lambda_2 3e-4 \
  --seed 42 \
  --log_dir runs/fair_nlg_xsum_ddp \
  --output_dir artifacts \
  --export_csv results_fair_nlg_xsum_ddp.csv \
  > logs/fair_nlg_xsum_ddp.out 2>&1 &

echo "Started fair XSum comparison (BART-large, all methods). Check logs/fair_nlg_xsum_ddp.out"
