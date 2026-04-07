#!/bin/bash
# ============================================================================
# 公平对比: GLUE 主线任务 × DeBERTa-v3-base × 全方法 (lora adalora evorank sora)
# ----------------------------------------------------------------------------
# 统一参数参考来源:
#   - SoRA 主线 no-schedule: lr=8e-4, bsz=8, warmup_ratio=0.06,
#     weight_decay=0.1, max_grad_norm=0.1, epochs=20, r=8,
#     sparse_lambda=10, sparse_lambda_2=3e-4, seed={0,21,42,81,100}
#
# 公平原则:
#   - 训练超参按 SoRA 官方 GLUE 主线统一
#   - 不再显式传 --target_modules，让各方法走各自论文/官方实现的默认注入协议
#   - 方法特有参数 (adalora_*, sora_*) 仅保留各自论文主线所需自由度
#
# EvoRank 专用（仅影响 method=evorank；其它方法忽略）:
#   --mini_val_k   从验证集 loader 取前 K 个 batch 缓存，供每轮 ES trial 上算 reward。
#                  过小则 reward 方差大、结构决策更噪；run_benchmark 默认 8。
#   --lambda_pop   每轮 ES 里最多评估的结构候选数（不含始终参与的 no-op）；
#                  配合默认 --population_strategy all 时取 generate_mutations 列表前 N 个。
#                  省略或设为 <=0 表示不截断（评估全部候选）。曾用 2 可省墙钟，但易漏掉更优剪枝/重分配。
#   --expand_init_mode  仅 evorank：zero=B 列 cold start；gradient=论文 Prop 3.2 梯度主方向初始化。
#                       同台五法可统一写上，其它方法忽略。
# ============================================================================
mkdir -p logs runs artifacts

nohup torchrun --nproc_per_node=2 --master_port=29500 \
  run_benchmark.py \
  --ddp \
  --methods lora adalora evorank sora toplora flatlora \
  --flatlora_rho 0.05 \
  --task_list mnli sst2 cola qqp qnli mrpc stsb \
  --model_list microsoft/deberta-v3-base \
  --target_rank 8 \
  --lora_alpha 16 \
  --epochs 20 \
  --batch_size 8 \
  --max_length 128 \
  --lr 8e-4 \
  --warmup_ratio 0.06 \
  --weight_decay 0.1 \
  --max_grad_norm 0.1 \
  --adalora_delta_t 100 \
  --adalora_orth_reg_weight 0.1 \
  --sora_sparse_lambda 10 \
  --sora_sparse_lambda_2 3e-4 \
  --lambda_c 0.001 \
  --expand_init_mode gradient \
  --evo_max_reallocate_candidates 8 \
  --verify_n_samples 0 \
  --seed_list 0 21 42 81 100 \
  --log_dir runs/fair_glue_deberta_ddp \
  --output_dir artifacts \
  --export_csv results_fair_glue_deberta_ddp.csv \
  > logs/fair_glue_deberta_ddp.out 2>&1 &

echo "Started fair GLUE comparison (DeBERTa-v3-base, all methods). Check logs/fair_glue_deberta_ddp.out"
