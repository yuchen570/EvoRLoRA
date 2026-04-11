#!/bin/bash
# ============================================================================
# 公平对比: XSum × BART-large × 全方法
# ============================================================================
#
# 超参来源:
#   - AdaLoRA 官方 NLG_QA/scripts/run_bart_xsum.sh:
#     lr=5e-4, epochs=25, per_device_bsz=8 (×8GPU=64 effective),
#     lora_alpha=32, target_modules=q_proj,k_proj,v_proj,out_proj,fc1,fc2,
#     max_source_length=768, max_target_length=64, warmup=3000 steps,
#     orth_reg=0.1, no weight_decay specified (默认 0)
#   - LoRA 原始 NLG (GPT-2): lr=2e-4, epochs=5, wd=0.01, alpha=32
#
# 本脚本使用 2 GPU, batch_size=32/GPU → effective=64 (与 AdaLoRA 8GPU×8 对齐)
# warmup: AdaLoRA 用 3000 steps, total ≈ 3156 steps/epoch × 25 ≈ 78900,
#          3000/78900 ≈ 0.04, 取 warmup_ratio=0.04
# max_grad_norm=1.0 (标准默认, 非 SoRA 专用 0.1)
#
# EvoRank 参数与 fair_glue_deberta.sh 对齐
# ============================================================================
mkdir -p logs runs artifacts

nohup torchrun --nproc_per_node=2 --master_port=29530 \
  run_benchmark.py \
  --ddp \
  --task_type nlg \
  --nlg_dataset_name xsum \
  --task_name xsum \
  --methods lora adalora evorank sora toplora flatlora pissa \
  --comparison_protocol controlled_fair \
  --protocol_dropout 0.05 \
  --module_preset custom \
  --flatlora_rho 0.05 \
  --model_name facebook/bart-large \
  --target_rank 8 \
  --lora_alpha 32 \
  --target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
  --epochs 25 \
  --batch_size 32 \
  --max_length 768 \
  --max_target_length 64 \
  --generation_max_new_tokens 64 \
  --lr 5e-4 \
  --warmup_ratio 0.04 \
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
  --log_dir runs/fair_nlg_xsum_ddp \
  --output_dir artifacts \
  --export_csv results_fair_nlg_xsum_ddp.csv \
  > logs/fair_nlg_xsum_ddp.out 2>&1 &

echo "Started fair XSum comparison (BART-large, all methods). Check logs/fair_nlg_xsum_ddp.out"
