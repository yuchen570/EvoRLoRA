#!/bin/bash
# ============================================================================
# CausalLM SFT 对比实验（对齐 PiSSA 官方 scripts/ 超参）
# ============================================================================
# 参考：PiSSA/scripts/python_llama2_7b/run_pissa.sh & run_lora.sh
#        PiSSA/scripts/metamath_llama2_7b/run_pissa.sh & run_lora.sh
#
# PiSSA 官方超参（8GPU, Llama-2-7b-hf）:
#   r=128, alpha=128, dropout=0, lr=2e-5, epochs=1
#   weight_decay=0, warmup_ratio=0.03, lr_scheduler=cosine
#   per_device_batch_size=4, gradient_accumulation_steps=4, 8GPUs → eff BS=128
#   model_max_length=512, optim=adamw_torch
#   target_modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
#
# 本脚本使用 2 GPU:
#   per_device_batch_size=4, gradient_accumulation_steps=16 → eff BS=4*16*2=128
#   与 PiSSA 8GPU 配置等效
#
# run_nlg_benchmark.py 已内置 cosine scheduler，与 PiSSA 一致
# ============================================================================

mkdir -p logs runs artifacts/nlg

MODELS=("models/meta-llama/Llama-2-7b-hf" "models/mistralai/Mistral-7B-v0.1" "models/google/gemma-7b")
TASKS=("metamath:100000" "python" "conversation")
METHODS=("lora" "lora_kaiming" "pissa" "evorank" "adalora" "sora" "flatlora" "toplora")
SEEDS=(42 123 321)

# 单模型、单任务、单种子冒烟测试
MODEL="models/meta-llama/Llama-2-7b-hf"
TASK="metamath:100000"
SEED=42

echo "Starting NLG Benchmark for Model: $MODEL on Task: $TASK"

for METHOD in "${METHODS[@]}"; do
    echo "==================================="
    echo "Running Method: $METHOD"
    echo "==================================="

    OUTPUT_DIR="artifacts/nlg/${MODEL}_${TASK}_${METHOD}_seed${SEED}"
    OUTPUT_DIR=${OUTPUT_DIR//\//_}
    OUTPUT_DIR=${OUTPUT_DIR//:/_}
    OUTPUT_DIR="artifacts/nlg/$OUTPUT_DIR"

    # 2 GPU × per_device_bs=4 × grad_accum=16 = Effective BS 128
    nohup torchrun --nproc_per_node=2 --master_port=29505 \
        run_nlg_benchmark.py \
        --model_name_or_path $MODEL \
        --method $METHOD \
        --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
        --sub_task $TASK \
        --epochs 1 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 16 \
        --learning_rate 2e-5 \
        --weight_decay 0.0 \
        --warmup_ratio 0.03 \
        --max_grad_norm 1.0 \
        --bf16 \
        --lora_rank 128 \
        --lora_alpha 128 \
        --lora_dropout 0.0 \
        --model_max_length 512 \
        --pissa_init_method pissa_niter_16 \
        --T_es 100 \
        --output_dir $OUTPUT_DIR \
        --seed $SEED \
        > logs/nlg_${METHOD}.out 2>&1 &

    echo "Started $METHOD. Check logs/nlg_${METHOD}.out"

    # 显存限制，串行等待每个方法跑完
    wait
done

echo "All tasks finished."
