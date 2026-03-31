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
# 方法特有参数 (adalora_*, sora_*, lora_ga_*) 仅作为该方法的内部自由度
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
  --batch_size 8 \
  --max_length 128 \
  --lr 8e-4 \
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
  --log_dir runs/fair_glue_deberta_ddp \
  --output_dir artifacts \
  --export_csv results_fair_glue_deberta_ddp.csv \
  > logs/fair_glue_deberta_ddp.out 2>&1 &

echo "Started fair GLUE comparison (DeBERTa-v3-base, all methods). Check logs/fair_glue_deberta_ddp.out"
