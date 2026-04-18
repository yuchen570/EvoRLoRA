#!/bin/bash
# ============================================================================
# CausalLM SFT + 下游评测（对齐 PiSSA 官方 Table 2: GSM8K/MATH/HumanEval/MBPP/MT-Bench）
# ============================================================================
# 训练超参参考：PiSSA/scripts/{metamath,python}_llama2_7b/run_pissa.sh
#   r=128, alpha=128, dropout=0, lr=2e-5, epochs=1,
#   weight_decay=0, warmup_ratio=0.03, cosine schedule
#   per_device_bs=4, grad_accum=4, 8GPU -> effective BS=128
#   model_max_length=512, adamw_torch
#   target_modules=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
#
# 本脚本使用 2GPU: per_device_bs=4, grad_accum=16 -> effective=128
# 每个方法训练结束后立即对 sub_task 的 test split 做生成评测并落 eval_results.json，
# 最终 scripts/summarize_nlg_pissa.py 会打印 Table 2 风格汇总。
# ============================================================================

set -e
mkdir -p logs runs artifacts/nlg

# 模型（可按需切换，首次运行会自动下载到 ./models）
MODEL="models/meta-llama/Llama-2-7b-hf"
# PiSSA 论文 sub_task 的通常组合：metamath (train) -> gsm8k+math (test);
# python (train) -> humaneval+mbpp (test); conversation (train) -> MT-Bench(单独)
TRAIN_SUB_TASK="metamath:100000"      # 训练集规模（遵循 PiSSA 100k）
EVAL_SUB_TASKS=(metamath python)      # test split 会自动带 type=gsm8k/math/humaneval/mbpp
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-0}   # 0 表示全量；调试时可设 200
USE_VLLM=${USE_VLLM:-0}                    # 1 时使用 vLLM，0 时用 HF generate
SEED=${SEED:-42}

METHODS=("lora" "lora_kaiming" "pissa" "evorank" "adalora" "sora" "flatlora" "toplora")

echo "============================================================="
echo "NLG benchmark (PiSSA Table 2 style)"
echo "  Model        : $MODEL"
echo "  Train task   : $TRAIN_SUB_TASK"
echo "  Eval tasks   : ${EVAL_SUB_TASKS[*]}"
echo "  Seed         : $SEED"
echo "  USE_VLLM     : $USE_VLLM"
echo "============================================================="

for METHOD in "${METHODS[@]}"; do
    echo "==================================="
    echo "[train] method=$METHOD"
    echo "==================================="

    TAG="${MODEL}_${TRAIN_SUB_TASK}_${METHOD}_seed${SEED}"
    TAG=${TAG//\//_}
    TAG=${TAG//:/_}
    OUTPUT_DIR="artifacts/nlg/${TAG}"

    # ------------------------------- 训练 -----------------------------------
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
        2>&1 | tee "logs/nlg_${METHOD}_train.log"

    # 无法合并成整模型的算法（adalora/evorank/sora/toplora）保存的是带 PEFT 结构的权重，
    # 不能直接被 AutoModelForCausalLM.from_pretrained 加载，因此下游生成评测暂时跳过。
    # lora / lora_kaiming / pissa / flatlora 已调用 merge_and_unload()，可直接加载。
    case "$METHOD" in
        lora|lora_kaiming|pissa|flatlora)
            EVAL_OK=1
            ;;
        *)
            EVAL_OK=0
            echo "[info] method=$METHOD 暂不支持 merge_and_unload 直接生成评测，已跳过下游评测。"
            ;;
    esac

    # ------------------------------- 评测 -----------------------------------
    if [ "$EVAL_OK" = "1" ]; then
        echo "==================================="
        echo "[eval]  method=$METHOD (sub_tasks=${EVAL_SUB_TASKS[*]})"
        echo "==================================="

        EVAL_FLAGS="--max_samples $EVAL_MAX_SAMPLES"
        if [ "$USE_VLLM" = "1" ]; then
            EVAL_FLAGS="$EVAL_FLAGS --use_vllm --tensor_parallel_size 2"
        fi

        python scripts/eval_nlg_pissa.py \
            --model_dir "$OUTPUT_DIR" \
            --sub_task "${EVAL_SUB_TASKS[@]}" \
            --dataset_split test \
            --method "$METHOD" \
            --model_tag "$(basename "$MODEL")" \
            --seed "$SEED" \
            --batch_size 8 \
            --max_new_tokens 512 \
            $EVAL_FLAGS \
            2>&1 | tee "logs/nlg_${METHOD}_eval.log"
    fi
done

# ------------------------- 汇总：PiSSA Table 2 风格 -----------------------
echo "============================================================="
echo "Summarizing results -> artifacts/nlg/table2_nlg_summary.md"
echo "============================================================="
python scripts/summarize_nlg_pissa.py \
    --root artifacts/nlg \
    --out_md artifacts/nlg/table2_nlg_summary.md

echo "All tasks finished."
