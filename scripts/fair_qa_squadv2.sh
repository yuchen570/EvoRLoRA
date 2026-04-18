#!/bin/bash
# ============================================================================
# 公平对比: SQuAD v2.0 × DeBERTa-v3-base × 全方法 × 4 档预算
#   对齐 AdaLoRA 论文 Table 2（v2 任务含不可答问题）
# ============================================================================

set -euo pipefail

mkdir -p logs runs artifacts artifacts/qa

MODEL=${MODEL:-"microsoft/deberta-v3-base"}
RANKS=${RANKS:-"1 2 4 8"}
METHODS=${METHODS:-"lora pissa adalora evorank sora flatlora toplora"}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-3}
BS=${BS:-16}
LR=${LR:-1e-3}
MAX_SEQ=${MAX_SEQ:-384}
DOC_STRIDE=${DOC_STRIDE:-128}
CSV=${CSV:-"results_fair_qa_squadv2.csv"}
NPROC=${NPROC:-2}
MASTER_PORT=${MASTER_PORT:-29561}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-0}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-0}

for method in ${METHODS}; do
  for r in ${RANKS}; do
    TAG="squadv2_deberta_${method}_r${r}_s${SEED}"
    OUTDIR="artifacts/qa/${TAG}"
    LOG="logs/${TAG}.out"
    echo "[${TAG}] launching..."
    torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT} \
      run_qa_benchmark.py \
      --model_name_or_path ${MODEL} \
      --method ${method} \
      --dataset_name squad_v2 \
      --version_2_with_negative \
      --max_seq_length ${MAX_SEQ} \
      --doc_stride ${DOC_STRIDE} \
      --num_train_epochs ${EPOCHS} \
      --max_train_steps ${MAX_TRAIN_STEPS} \
      --per_device_train_batch_size ${BS} \
      --per_device_eval_batch_size 64 \
      --learning_rate ${LR} \
      --weight_decay 0.0 \
      --warmup_ratio 0.1 \
      --max_grad_norm 1.0 \
      --lora_rank ${r} \
      --lora_alpha $((r * 2)) \
      --lora_dropout 0.05 \
      --adalora_tinit 500 \
      --adalora_tfinal 2500 \
      --adalora_delta_t 10 \
      --adalora_orth_reg_weight 0.1 \
      --sora_sparse_lambda 1e-3 \
      --flatlora_rho 0.05 \
      --max_train_samples ${MAX_TRAIN_SAMPLES} \
      --max_eval_samples ${MAX_EVAL_SAMPLES} \
      --output_dir "${OUTDIR}" \
      --export_csv "${CSV}" \
      --seed ${SEED} \
      > "${LOG}" 2>&1 || {
        echo "[${TAG}] FAILED, see ${LOG}"
        continue
      }
    echo "[${TAG}] done → ${OUTDIR} (${LOG})"
  done
done

echo ""
echo "All SQuAD v2.0 runs completed."
echo "Render AdaLoRA-style Table 2 with:"
echo "  python scripts/generate_qa_table.py \\"
echo "        --csv ${CSV} --task squad_v2 --out_md artifacts/qa/table2_squadv2.md"
