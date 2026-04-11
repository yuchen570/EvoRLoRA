#!/bin/bash
# ============================================================================
# 公平对比: CNN/DailyMail × BART-large × 全方法
# ============================================================================
#
# 超参来源:
#   - AdaLoRA 官方 NLG_QA/scripts/run_bart_cnndailymail.sh:
#     lr=5e-4, epochs=15, per_device_bsz=4 (×8GPU=32 effective),
#     lora_alpha=32, target_modules=q_proj,k_proj,v_proj,out_proj,fc1,fc2,
#     max_source_length=1024, max_target_length=160, warmup=3000 steps,
#     orth_reg=0.1, no weight_decay specified (默认 0)
#   - LoRA 原始 NLG (GPT-2): lr=2e-4, epochs=5, wd=0.01, alpha=32
#
# 本脚本使用 2 GPU, batch_size=16/GPU → effective=32 (与 AdaLoRA 8GPU×4 对齐)
# warmup: AdaLoRA 用 3000 steps, total ≈ 9375 steps/epoch × 15 ≈ 140625,
#          3000/140625 ≈ 0.02, 取 warmup_ratio=0.02
# max_grad_norm=1.0 (标准默认, 非 SoRA 专用 0.1)
#
# EvoRank 参数与 fair_glue_deberta.sh 对齐
# ============================================================================
mkdir -p logs runs artifacts

nohup torchrun --nproc_per_node=2 --master_port=29520 \
  run_benchmark.py \
  --ddp \
  --task_type nlg \
  --nlg_dataset_name cnn_dailymail \
  --task_name cnn_dailymail \
  --methods lora adalora evorank sora toplora flatlora pissa \
  --comparison_protocol controlled_fair \
  --protocol_dropout 0.05 \
  --module_preset custom \
  --flatlora_rho 0.05 \
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
  --warmup_ratio 0.02 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --adalora_delta_t 100 \
  --adalora_orth_reg_weight 0.1 \
  --sora_sparse_lambda 1e-3 \
  --sora_sparse_lambda_2 1e-4 \
  --lambda_c 0.0 \
  --expand_init_mode gradient \
  --mini_val_k 8 \
  --evo_alpha_u 1.0 \
  --evo_p_p 0.05 \
  --evo_H_p 4 \
  --evo_max_reallocate_candidates 16 \
  --seed 42 \
  --verify_n_samples 0 \
  --log_dir runs/fair_nlg_cnndm_ddp \
  --output_dir artifacts \
  --export_csv results_fair_nlg_cnndm_ddp.csv \
  > logs/fair_nlg_cnndm_ddp.out 2>&1 &

echo "Started fair CNN/DailyMail comparison (BART-large, all methods). Check logs/fair_nlg_cnndm_ddp.out"
