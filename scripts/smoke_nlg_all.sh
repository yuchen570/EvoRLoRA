#!/bin/bash
set -euo pipefail

# ==============================================================================
# Linux Smoke Test Script for ALL Methods and ALL Tasks
#
# 该脚本在 Linux 环境下走通 8 种算法的（前向/反向/保存/评估）冒烟测试，
# 不做性能测试，仅极小规模：
# - 训练：4行样本, 1 epoch, r=8
# - 评估：4个样本 (NLG) 或第1题 (MT-Bench)
#
# PEFT 方法 (lora/lora_kaiming/pissa/adalora/flatlora) 默认仅保存 adapter 权重
# (~10MB)，跳过 merge + 全量写盘 (~13GB/7B)，大幅加速冒烟循环。eval 脚本会自动
# 检测 adapter-only 保存并在内存中重建完整模型。
#
# 使用方法：
#   conda activate evorank
#   bash scripts/smoke_nlg_all.sh
#
# 可选环境变量：
#   MODEL_REL   模型相对/绝对路径  (默认 models/meta-llama/Llama-2-7b-hf)
#   MODEL_TAG   模型标签           (默认 Llama-2-7b-hf)
# ==============================================================================

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_REL="${MODEL_REL:-models/meta-llama/Llama-2-7b-hf}"
MODEL_TAG="${MODEL_TAG:-Llama-2-7b-hf}"
SEED=42

SMOKE_ROWS=4
EVAL_MAX_SAMPLES=4

METHODS=("lora" "lora_kaiming" "pissa" "evorank" "adalora" "sora" "flatlora" "toplora")
TASKS=("metamath" "python" "conversation")

mkdir -p logs artifacts

get_method_extra_args() {
    local method=$1
    if [ "$method" == "adalora" ]; then
        echo "--adalora_delta_t 1 --adalora_tinit 0 --adalora_tfinal 8"
    elif [ "$method" == "evorank" ]; then
        echo "--T_es 4 --mini_val_k 1"
    fi
}

FAIL_COUNT=0
FAIL_LIST=""

for method in "${METHODS[@]}"; do
    for task in "${TASKS[@]}"; do
        out_dir="${REPO_ROOT}/artifacts/nlg_smoke_${MODEL_TAG}_${task}_${method}_seed${SEED}"
        extra_args=$(get_method_extra_args "$method")

        echo -e "\n\033[1;36m========== [train] method=$method task=$task -> $out_dir ==========\033[0m"
        train_log="logs/smoke_${MODEL_TAG}_${task}_${method}_train.log"

        if ! python run_nlg_benchmark.py \
            --model_name_or_path "$MODEL_REL" \
            --method "$method" \
            --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
            --sub_task "${task}:${SMOKE_ROWS}" \
            --epochs 1 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 1 \
            --learning_rate 2e-4 \
            --weight_decay 0.0 \
            --warmup_ratio 0.03 \
            --max_grad_norm 1.0 \
            --bf16 \
            --lora_rank 8 \
            --lora_alpha 16 \
            --lora_dropout 0.0 \
            --model_max_length 128 \
            --pissa_init_method "pissa_niter_16" \
            --T_es 10000 \
            --save_adapter_only \
            --output_dir "$out_dir" \
            --seed "$SEED" $extra_args 2>&1 | tee "$train_log"; then
            echo -e "\033[1;31m[FAIL] train $method/$task\033[0m"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            FAIL_LIST="${FAIL_LIST}\n  train ${method}/${task}"
            continue
        fi

        if [ ! -d "$out_dir" ]; then
            echo -e "\033[1;31m[FAIL] train $method/$task: output dir not created\033[0m"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            FAIL_LIST="${FAIL_LIST}\n  train ${method}/${task} (no output dir)"
            continue
        fi

        if [ "$task" == "conversation" ]; then
            echo -e "\033[1;32m========== [eval] mtbench $task method=$method ==========\033[0m"
            eval_log="logs/smoke_${MODEL_TAG}_${task}_${method}_eval.log"
            if ! python scripts/eval_mtbench.py \
                --model_dir "$out_dir" \
                --method "$method" \
                --model_tag "$MODEL_TAG" \
                --seed "$SEED" \
                --max_new_tokens 256 \
                --num_gpus 1 \
                --question-begin 0 \
                --question-end 1 2>&1 | tee "$eval_log"; then
                echo -e "\033[1;31m[FAIL] eval $method/$task\033[0m"
                FAIL_COUNT=$((FAIL_COUNT + 1))
                FAIL_LIST="${FAIL_LIST}\n  eval ${method}/${task}"
            fi
        else
            echo -e "\033[1;32m========== [eval] nlg $task method=$method ==========\033[0m"
            eval_log="logs/smoke_${MODEL_TAG}_${task}_${method}_eval.log"
            if ! python scripts/eval_nlg_pissa.py \
                --model_dir "$out_dir" \
                --sub_task "$task" \
                --dataset_split test \
                --method "$method" \
                --model_tag "$MODEL_TAG" \
                --seed "$SEED" \
                --batch_size 2 \
                --max_new_tokens 128 \
                --max_samples "$EVAL_MAX_SAMPLES" 2>&1 | tee "$eval_log"; then
                echo -e "\033[1;31m[FAIL] eval $method/$task\033[0m"
                FAIL_COUNT=$((FAIL_COUNT + 1))
                FAIL_LIST="${FAIL_LIST}\n  eval ${method}/${task}"
            fi
        fi
    done
done

echo ""
echo "======================================================================"
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "\033[1;33mSmoke: All ${#METHODS[@]} methods × ${#TASKS[@]} tasks finished OK.\033[0m"
else
    echo -e "\033[1;31mSmoke: ${FAIL_COUNT} step(s) FAILED:${FAIL_LIST}\033[0m"
    exit 1
fi
