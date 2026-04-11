#!/bin/bash
# ============================================================================
# 公平对比: GLUE × DeBERTa-v3-base × 全方法
# ============================================================================
#
# 1. AdaLoRA 官方仓库 (ICLR 2023) 提供了 DeBERTa-v3-base 在全部 8 个 GLUE
#    子任务上的独立超参，是唯一覆盖面最全的基准。
# 2. SoRA 官方仓库 (EMNLP 2023) 自己的 LoRA 基线使用 lr=3e-4 / epochs=10
#    （而非 SoRA 的 8e-4/20），验证了高 LR+长训适配 SoRA 但不适配 LoRA。
# 3. LoRA 原始仓库用 DeBERTa-v2-XXLarge，lr 极低 (6e-5~1e-4)，
#    DeBERTa-v3-base 体量更小可适当提高。
#
# 结论：按数据集规模分两挡，以 AdaLoRA 超参为主基准：
#   大数据集 (MNLI/SST2/QQP/QNLI): lr=5e-4, epochs=7
#   小数据集 (CoLA/RTE/MRPC/STS-B): lr=8e-4, epochs=25
#
# 方法特有参数保持各自论文默认。SoRA 使用论文专用 max_grad_norm=0.1，
# 其他方法使用标准 max_grad_norm=1.0。
# ============================================================================

mkdir -p logs runs artifacts

# ===========================================================================
#  大数据集子任务 (样本>60k): MNLI, SST-2, QQP, QNLI
# ===========================================================================
nohup torchrun --nproc_per_node=2 --master_port=29500 \
  run_benchmark.py \
  --ddp \
  --methods lora adalora evorank sora toplora flatlora pissa \
  --comparison_protocol controlled_fair \
  --protocol_dropout 0.05 \
  --module_preset default \
  --flatlora_rho 0.05 \
  --task_list mnli sst2 qqp qnli \
  --model_list microsoft/deberta-v3-base \
  --target_rank 8 \
  --lora_alpha 16 \
  --epochs 7 \
  --batch_size 32 \
  --max_length 256 \
  --lr 5e-4 \
  --warmup_ratio 0.06 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --adalora_delta_t 100 \
  --adalora_orth_reg_weight 0.1 \
  --sora_sparse_lambda 10 \
  --sora_sparse_lambda_2 3e-4 \
  --lambda_c 0.0 \
  --expand_init_mode gradient \
  --mini_val_k 8 \
  --evo_alpha_u 1.0 \
  --evo_p_p 0.05 \
  --evo_H_p 4 \
  --evo_max_reallocate_candidates 16 \
  --verify_n_samples 0 \
  --seed_list 0 21 42 81 100 \
  --log_dir runs/fair_glue_deberta_large_ddp \
  --output_dir artifacts \
  --export_csv results_fair_glue_deberta_large_ddp.csv \
  > logs/fair_glue_deberta_large_ddp.out 2>&1 &

echo "Started LARGE-task benchmark (MNLI/SST2/QQP/QNLI). Check logs/fair_glue_deberta_large_ddp.out"

# ===========================================================================
#  小数据集子任务 (样本<10k): CoLA, RTE, MRPC, STS-B
# ===========================================================================
nohup torchrun --nproc_per_node=2 --master_port=29501 \
  run_benchmark.py \
  --ddp \
  --methods lora adalora evorank sora toplora flatlora pissa \
  --comparison_protocol controlled_fair \
  --protocol_dropout 0.05 \
  --module_preset default \
  --flatlora_rho 0.05 \
  --task_list cola rte mrpc stsb \
  --model_list microsoft/deberta-v3-base \
  --target_rank 8 \
  --lora_alpha 32 \
  --epochs 25 \
  --batch_size 32 \
  --max_length 128 \
  --lr 8e-4 \
  --warmup_ratio 0.06 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --adalora_delta_t 10 \
  --adalora_orth_reg_weight 0.1 \
  --sora_sparse_lambda 10 \
  --sora_sparse_lambda_2 3e-4 \
  --lambda_c 0.0 \
  --expand_init_mode gradient \
  --mini_val_k 8 \
  --evo_alpha_u 1.0 \
  --evo_p_p 0.05 \
  --evo_H_p 4 \
  --evo_max_reallocate_candidates 16 \
  --verify_n_samples 0 \
  --seed_list 0 21 42 81 100 \
  --log_dir runs/fair_glue_deberta_small_ddp \
  --output_dir artifacts \
  --export_csv results_fair_glue_deberta_small_ddp.csv \
  > logs/fair_glue_deberta_small_ddp.out 2>&1 &

echo "Started SMALL-task benchmark (CoLA/RTE/MRPC/STS-B). Check logs/fair_glue_deberta_small_ddp.out"
