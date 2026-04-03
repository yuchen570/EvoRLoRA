#!/bin/bash
# ============================================================================
# 公平对比: GLUE 9 项 × DeBERTa-v3-base × 全方法 (lora adalora evorank lora-ga sora)
# ----------------------------------------------------------------------------
# 统一参数参考来源:
#   - SoRA (run_glue_sora_no_schedule.sh): lr=8e-4, bsz=8, warmup_ratio=0.06,
#     weight_decay=0.1, max_grad_norm=0.1, epochs=20, r=8, seed={0,21,42,81,100}
#   - AdaLoRA (NLU/scripts): lora_module=query,key,value,intermediate,...,attention.output
#     → DeBERTa-v3 PEFT 映射: query_proj,key_proj,value_proj,intermediate.dense,output.dense
#   - LoRA-GA (reproduce): lora_alpha=16, stable_gamma=16, use_rslora=false
#
# 公平原则: 同 lr / epochs / batch_size / warmup / weight_decay / grad_norm / seed
# 注：为避免 LoRA 在 DeBERTa+CoLA 上出现 NaN，本脚本统一使用更稳定的 2e-4。
# 方法特有参数 (adalora_*, sora_*, lora_ga_*) 仅作为该方法的内部自由度
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
  --methods lora adalora evorank lora-ga sora \
  --task_list cola sst2 mrpc qqp stsb mnli qnli wnli \
  --model_list microsoft/deberta-v3-base \
  --target_rank 8 \
  --lora_alpha 16 \
  --target_modules query_proj,key_proj,value_proj,intermediate.dense,output.dense \
  --epochs 20 \
  --batch_size 32 \
  --max_length 128 \
  --lr 2e-4 \
  --warmup_ratio 0.06 \
  --weight_decay 0.1 \
  --max_grad_norm 0.1 \
  --adalora_delta_t 100 \
  --adalora_orth_reg_weight 0.1 \
  --lora_ga_batches 8 \
  --lora_ga_stable_gamma 16 \
  --sora_sparse_lambda 10 \
  --sora_sparse_lambda_2 3e-4 \
  --expand_init_mode gradient \
  --seed_list 0 21 42 81 100 \
  --log_dir runs/fair_glue_deberta_ddp \
  --output_dir artifacts \
  --export_csv results_fair_glue_deberta_ddp.csv \
  > logs/fair_glue_deberta_ddp.out 2>&1 &

echo "Started fair GLUE comparison (DeBERTa-v3-base, all methods). Check logs/fair_glue_deberta_ddp.out"
