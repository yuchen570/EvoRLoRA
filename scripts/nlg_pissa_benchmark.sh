#!/bin/bash
# ============================================================================
# CausalLM SFT + 下游评测（对齐 PiSSA 官方 Table 2）
# ============================================================================
# Table 2 完整实验矩阵:
#   模型:   LLaMA-2-7B, Mistral-7B, Gemma-7B
#   训练:   metamath:100000 → 评 GSM8K + MATH
#           python          → 评 HumanEval + MBPP
#           conversation    → 评 MT-Bench
#   方法:   lora, lora_kaiming, pissa, evorank, adalora, sora, flatlora, toplora
#
# 训练超参参考: PiSSA/scripts/{metamath,python,conveision}_llama2_7b/run_pissa.sh
#   r=128, alpha=128, dropout=0, lr=2e-5, epochs=1,
#   weight_decay=0, warmup_ratio=0.03, cosine schedule
#   per_device_bs=4, grad_accum=4, 8GPU -> effective BS=128
#   model_max_length=512, adamw_torch
#   target_modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
#
# 本脚本使用 2GPU: per_device_bs=8, grad_accum=8 -> effective=128
#
# 流程分为两阶段:
#   阶段 1: 对所有 模型×任务×方法 依次训练，保存合并后的完整模型
#   阶段 2: 所有训练完成后统一进行下游生成评测
# 最终 scripts/summarize_nlg_pissa.py 会打印 Table 2 风格汇总。
# ============================================================================

set -euo pipefail
mkdir -p logs runs artifacts/nlg

# ========================= 配置 =========================
MODELS=(
    "models/meta-llama/Llama-2-7b-hf"
    "models/mistralai/Mistral-7B-v0.1"
    "models/google/gemma-7b"
)

# 训练子任务 → 对应的评测子任务
# metamath:100000 训练 → metamath test (含 type=gsm8k/math)
# python 训练          → python test (含 type=humaneval/mbpp)
# conversation 训练    → MT-Bench (独立评测)
TRAIN_TASKS=("metamath:100000" "python" "conversation")
EVAL_TASK_MAP_metamath="metamath"
EVAL_TASK_MAP_python="python"
EVAL_TASK_MAP_conversation="mt_bench"

METHODS=("lora" "lora_kaiming" "pissa" "evorank" "adalora" "sora" "flatlora" "toplora")

EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-0}   # 0 表示全量；调试时可设 200
USE_VLLM=${USE_VLLM:-0}                    # 1 时使用 vLLM，0 时用 HF generate
SEED=${SEED:-42}

echo "============================================================="
echo "NLG benchmark (PiSSA Table 2 — full matrix)"
echo "  Models       : ${MODELS[*]}"
echo "  Train tasks  : ${TRAIN_TASKS[*]}"
echo "  Methods      : ${METHODS[*]}"
echo "  Seed         : $SEED"
echo "  USE_VLLM     : $USE_VLLM"
echo "============================================================="

# ========================= 阶段 1: 训练所有组合 =========================
# 使用文件记录每个组合的输出目录，供评测阶段读取
TRAIN_MANIFEST="artifacts/nlg/train_manifest_seed${SEED}.txt"
> "$TRAIN_MANIFEST"

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(basename "$MODEL")
    for TRAIN_SUB_TASK in "${TRAIN_TASKS[@]}"; do
        # 提取训练任务名称（去掉 :数量 后缀）
        TASK_NAME="${TRAIN_SUB_TASK%%:*}"

        for METHOD in "${METHODS[@]}"; do
            echo "==================================="
            echo "[train] model=$MODEL_SHORT  task=$TRAIN_SUB_TASK  method=$METHOD"
            echo "==================================="

            TAG="${MODEL_SHORT}_${TRAIN_SUB_TASK}_${METHOD}_seed${SEED}"
            TAG=${TAG//\//_}
            TAG=${TAG//:/_}
            OUTPUT_DIR="$(pwd)/artifacts/nlg/${TAG}"
            LOG_TAG="nlg_${MODEL_SHORT}_${TASK_NAME}_${METHOD}"

            torchrun --nproc_per_node=2 --master_port=29505 \
                run_nlg_benchmark.py \
                --model_name_or_path "$MODEL" \
                --method "$METHOD" \
                --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
                --sub_task "$TRAIN_SUB_TASK" \
                --epochs 1 \
                --per_device_train_batch_size 8 \
                --gradient_accumulation_steps 8 \
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
                --output_dir "$OUTPUT_DIR" \
                --seed "$SEED" \
                2>&1 | tee "logs/${LOG_TAG}_train.log"

            # 记录: MODEL_SHORT|TASK_NAME|METHOD|OUTPUT_DIR
            echo "${MODEL_SHORT}|${TASK_NAME}|${METHOD}|${OUTPUT_DIR}" >> "$TRAIN_MANIFEST"
        done
    done
done

echo "============================================================="
echo "All training completed. Starting evaluation..."
echo "============================================================="

# ========================= 阶段 2: 统一评测 =========================
EVAL_FLAGS="--max_samples $EVAL_MAX_SAMPLES"
if [ "$USE_VLLM" = "1" ]; then
    EVAL_FLAGS="$EVAL_FLAGS --use_vllm --tensor_parallel_size 2"
fi

while IFS='|' read -r MODEL_SHORT TASK_NAME METHOD OUTPUT_DIR; do
    echo "==================================="
    echo "[eval]  model=$MODEL_SHORT  task=$TASK_NAME  method=$METHOD"
    echo "==================================="

    LOG_TAG="nlg_${MODEL_SHORT}_${TASK_NAME}_${METHOD}"

    if [ "$TASK_NAME" = "conversation" ]; then
        # MT-Bench: 使用 FastChat 评测
        python scripts/eval_mtbench.py \
            --model_dir "$OUTPUT_DIR" \
            --method "$METHOD" \
            --model_tag "$MODEL_SHORT" \
            --seed "$SEED" \
            --max_new_tokens 1024 \
            2>&1 | tee "logs/${LOG_TAG}_eval.log"
    else
        # metamath → gsm8k/math;  python → humaneval/mbpp
        python scripts/eval_nlg_pissa.py \
            --model_dir "$OUTPUT_DIR" \
            --sub_task "$TASK_NAME" \
            --dataset_split test \
            --method "$METHOD" \
            --model_tag "$MODEL_SHORT" \
            --seed "$SEED" \
            --batch_size 8 \
            --max_new_tokens 512 \
            $EVAL_FLAGS \
            2>&1 | tee "logs/${LOG_TAG}_eval.log"
    fi
done < "$TRAIN_MANIFEST"

# ------------------------- 汇总：PiSSA Table 2 风格 -----------------------
echo "============================================================="
echo "Summarizing results -> artifacts/nlg/table2_nlg_summary.md"
echo "============================================================="
python scripts/summarize_nlg_pissa.py \
    --root artifacts/nlg \
    --out_md artifacts/nlg/table2_nlg_summary.md

echo "All tasks finished."
