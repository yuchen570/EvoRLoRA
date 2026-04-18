#!/bin/bash
# ============================================================================
# 公平对比: SQuAD v1.1 × DeBERTa-v3-base × 全方法 × 4 档预算
#   对齐 AdaLoRA 论文 Table 2
# ============================================================================
# 预算档位（对齐论文 b^{(T)} 列）：
#   0.08%  → lora_rank=1   (r·2·6·12 ≈ 144 → total params share per module ~ 1)
#   0.16%  → lora_rank=2
#   0.32%  → lora_rank=4
#   0.65%  → lora_rank=8
#
# 超参来源: AdaLoRA/NLG_QA/scripts/run_debertav3_squadv1.sh（lr=1e-3, bsz=16, epochs=10,
#   max_seq_length=384, doc_stride=128, reg_orth_coef=0.1,
#   init_warmup=5000, final_warmup=25000, mask_interval=100）。
# 这里收敛到 3 epoch + 单机 2GPU 以便本机跑通；需要完整复现时把 EPOCHS=10, RANKS 保持不变。
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
CSV=${CSV:-"results_fair_qa_squadv1.csv"}
NPROC=${NPROC:-2}
MASTER_PORT=${MASTER_PORT:-29560}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-0}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-0}

for method in ${METHODS}; do
  for r in ${RANKS}; do
    TAG="squadv1_deberta_${method}_r${r}_s${SEED}"
    OUTDIR="artifacts/qa/${TAG}"
    LOG="logs/${TAG}.out"
    echo "[${TAG}] launching..."
    torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT} \
      run_qa_benchmark.py \
      --model_name_or_path ${MODEL} \
      --method ${method} \
      --dataset_name squad \
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
echo "All SQuAD v1.1 runs completed."
echo "Render AdaLoRA-style Table 2 with:"
echo "  python scripts/generate_qa_table.py \\"
echo "        --csv ${CSV} --task squad --out_md artifacts/qa/table2_squadv1.md"
