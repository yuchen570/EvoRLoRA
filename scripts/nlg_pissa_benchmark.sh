#!/bin/bash
# ============================================================================
# CausalLM SFT 对比实验 (PiSSA Table 2)
# ============================================================================
# 模型：LLaMA-2-7B, Mistral-7B, Gemma-7B
# 数据集：metamath:100000, python, conversation
# 训练超参：r=128, alpha=128, dropout=0, lr=2e-5, epochs=1
# Effective Batch Size = 128 (4 * 4 * 8GPUs)
# ============================================================================

mkdir -p logs runs artifacts/nlg

MODELS=("models/meta-llama/Llama-2-7b-hf" "models/mistralai/Mistral-7B-v0.1" "models/google/gemma-7b")
TASKS=("metamath:100000" "python" "conversation")
METHODS=("lora" "lora_kaiming" "pissa" "evorank" "adalora" "sora" "flatlora" "toplora")
SEEDS=(42 123 321)

# === 如果你想在 1 台拥有 8 个 GPU 的机器上跑完整评测，循环全部即可 ===
# 但是单次耗时非常久，建议先使用单模型、单任务、单种子跑一遍冒烟测试：

MODEL="models/meta-llama/Llama-2-7b-hf"
TASK="metamath:100000"
SEED=42

echo "Starting NLG Benchmark for Model: $MODEL on Task: $TASK"

for METHOD in "${METHODS[@]}"; do
    echo "==================================="
    echo "Running Method: $METHOD"
    echo "==================================="
    
    OUTPUT_DIR="artifacts/nlg/${MODEL}_${TASK}_${METHOD}_seed${SEED}"
    # 防止路径包含特殊字符
    OUTPUT_DIR=${OUTPUT_DIR//\//_}
    OUTPUT_DIR=${OUTPUT_DIR//:/_}
    OUTPUT_DIR="artifacts/nlg/$OUTPUT_DIR"

    # 使用 torchrun (支持 DeepSpeed 或单纯的 DDP)
    # 本脚本以单机 双卡 为例，单卡批次 4，梯度累加 16 => 总 Effective BS 128
    # 若环境 GPU 不足，请自行修改 --nproc_per_node 和 --gradient_accumulation_steps
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
        --lora_rank 128 \
        --lora_alpha 128 \
        --lora_dropout 0.0 \
        --T_es 100 \
        --output_dir $OUTPUT_DIR \
        --seed $SEED \
        > logs/nlg_${METHOD}.out 2>&1 &
        
    echo "Started $METHOD. Check logs/nlg_${METHOD}.out"
    
    # 因为显存会被全部占用，串行等待每个方法跑完。
    wait
done

echo "All tasks finished."
