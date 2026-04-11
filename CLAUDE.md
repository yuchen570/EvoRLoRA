# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EvoRLoRA is a unified benchmark harness for 7 LoRA variants, with **EvoRank-LoRA** as the core contribution — an evolutionary rank-adaptive LoRA that dynamically adjusts per-layer rank during training using evolutionary strategies (ES).

Supported `--methods`: `lora`, `adalora`, `pissa`, `evorank`, `sora`, `flatlora`, `toplora`

## Environment Setup

```bash
conda create -y -p ./envs/evorank python=3.10
conda activate ./envs/evorank
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

HuggingFace datasets/models cache to `./datasets` and `./models` by default.

## Running Experiments

**Smoke test (single GPU, ~1 min):**
```bash
python run_benchmark.py \
  --methods lora adalora evorank \
  --task_name sst2 --model_name roberta-base \
  --max_train_steps 50 --T_es 20 \
  --adalora_delta_t 50 --adalora_orth_reg_weight 0.1 \
  --expand_init_mode gradient --seed 42 \
  --output_dir artifacts --export_csv results_smoke.csv
```

**Full GLUE benchmark (canonical, from `scripts/fair_glue_deberta.sh`):**
```bash
nohup torchrun --nproc_per_node=2 --master_port=29500 run_benchmark.py \
  --ddp \
  --methods lora adalora evorank sora toplora flatlora pissa \
  --task_list cola sst2 mrpc qqp stsb mnli qnli rte wnli \
  --model_list microsoft/deberta-v3-base \
  --target_rank 8 --lora_alpha 16 --epochs 20 \
  --batch_size 8 --lr 2e-4 --weight_decay 0.1 \
  --warmup_ratio 0.06 --max_grad_norm 0.1 \
  --seed_list 0 21 42 81 100 \
  --expand_init_mode gradient \
  --output_dir artifacts --export_csv results_main.csv \
  > logs/main.out 2>&1 &
```

RTE uses a separate script (`scripts/fair_glue_deberta_rte.sh`) aligned with the main fair protocol (`epochs=20, batch_size=8, lr=8e-4`; see script).

**Ablation studies:**
```bash
bash scripts/ablate_evorank_glue_deberta.sh
bash scripts/ablate_evorank_reallocation_efficiency.sh
python scripts/summarize_evorank_ablation.py
python scripts/summarize_evorank_reallocation_efficiency.py
```

**NLG summarization benchmarks:**
```bash
bash scripts/fair_nlg_cnndailymail.sh   # BART-large × CNN/DailyMail, 2GPU
bash scripts/fair_nlg_xsum.sh           # BART-large × XSum, 2GPU
```

**Generate paper-format GLUE table:**
```bash
python scripts/generate_glue_table.py   # reads results_fair_glue_deberta_*.csv from cwd
```

## Architecture

### Core Files

| File | Role |
|------|------|
| `evo_rank_lora.py` | `EvoRankLoRALayer`: rank super-space (r_max), boolean active mask, gradient caching for ES |
| `rank_evolution_controller.py` | `RankEvolutionController`: expand/prune/reallocate mutations, EMA statistics, dynamic thresholds, cooldown |
| `train_integration.py` | `inject_evo_lora()` model injection; `train_evo_lora_step()` dual time-scale loop |
| `run_benchmark.py` | Main entry point (2558 lines): all CLI args, GLUE+NLG data loading, training loop, CSV export |
| `run_nlg_benchmark.py` | NLG entry point: SFT on instruction-following datasets; single `--method` per invocation (not a multi-method sweep); targets CausalLM (Llama-2-7b default); no CSV export, different CLI conventions from `run_benchmark.py` |
| `glue_metrics.py` | Task-specific metrics: Matthews (CoLA), F1 (MRPC/QQP), Pearson+Spearman mean (STS-B), Accuracy (rest) |
| `sora_inject.py` | `SoRALinear` (per-rank soft-threshold gate); `SparseAdamW`: full AdamW update with proximal soft-thresholding on gate params (threshold = `sparse_lambda`, not `lr × sparse_lambda`); supports `linear`/`log_linear`/`exp_linear` lambda schedules |
| `toplora_inject.py` | `TopSingularValue`: token-wise scaling `λ(x) = exp(RMSNorm(x @ W_λ))`, `W_λ ∈ R^{d_in × r}`, Kaiming fan_out init; applied between lora_A and lora_B projections; extra param cost = `d_in × r` per layer |
| `flatlora_inject.py` | `FlatLoRAHookManager`: cosine-increasing noise schedule `factor = 0.5(1 − cos(πt/T))`; per-row norm scaling; noise added in `forward_pre_hook`, subtracted deterministically in `full_backward_hook` via stored seed — O(out_features) memory |
| `adalora_utils.py` | Orthogonal regularization loss + rank budget update helpers |
| `configs/ds_config_zero2_no_offload.json` | DeepSpeed ZeRO-2 (bf16, no CPU offload, `overlap_comm=True`); used by `run_nlg_benchmark.py` with large CausalLMs; not used by `run_benchmark.py` (native DDP) |

### Reference Implementations (not part of harness)

The following subdirectories contain original paper source code for algorithm verification only. Do not import from them in new code.

| Directory | Source |
|-----------|--------|
| `AdaLoRA/` | Official AdaLoRA repo (NLU/ and NLG_QA/ branches) |
| `LoRA/` | Original LoRA loralib |
| `PiSSA/` | PiSSA training scripts and utilities |
| `SoRA/` | TsinghuaC3I/SoRA reference |
| `Flat-LoRA/` | Flat-LoRA ICML 2025 reference |
| `toplora-neurips25/` | TopLoRA NeurIPS 2025 reference |

### Dual Time-Scale Training (EvoRank)

1. **Fast loop** (every step): gradient descent on LoRA weights; gradient stats cached *before* `clip_grad_norm_`
2. **Slow loop** (every `--T_es` steps): generate candidate mutations → evaluate each on mini validation set → commit best mutation; trial rewards aggregated via `all_reduce` in DDP

### Output Artifacts

`artifacts/<task>_<backbone>_<method>/`:
- `metrics.jsonl` — per-step metrics
- `final/` — adapter weights + tokenizer (self-contained for inference)
- `checkpoint_epoch_*.pt` — optional intermediate checkpoints (with `--save_every_epoch`)

CSV output aggregates across seeds with mean/std rows per `task×backbone×method`.

## Key Gotchas

- **`ax` task is unsupported** — HuggingFace GLUE has 10 configs but `ax` has no gold labels; `run_benchmark.py` will explicitly error if `ax` is in `--task_list`. Use the 9 supervised tasks only.
- **Epoch recommendations differ by task**: large datasets (sst2, mnli, qnli, qqp) converge in 3–5 epochs; small datasets (cola, mrpc, stsb, rte, wnli) need 15–20 epochs.
- **`--head_lr`**: classifier/score heads are automatically pushed to `max(lr, 5e-4)` to prevent random-init heads from failing to converge on CoLA/MRPC.
- **AdaLoRA `--adalora_delta_t`**: default is `0.1 * total_steps` (not 0.8) so AdaLoRA has 90% of training for dynamic rank adjustment.
- **SoRA `--sora_sparse_lambda_2`**: default `3e-4` (not `1e-3`) to prevent aggressive pruning mid-training.
- **`--expand_init_mode gradient`**: uses projected gradient principal singular vector (Proposition 3.2) for new rank-1 slot initialization; `zero` is cold-start. Only affects `evorank`; safe to pass for all-method runs.
- **DDP**: validation metrics computed on rank0 over full dev set, then broadcast. Artifacts written by rank0 only.
- **`--evo_max_reallocate_candidates 8`**: limits cross-layer reallocation candidates to prevent combinatorial explosion; set to `0` for unlimited (ablation only).
- **EvoRank warmup cap**: Both LR warmup and ES warmup are capped to `min(warmup_steps, max(20, 10% × steps_per_epoch))`. With LoRA B=0 zero-init, a long warmup (e.g., 6% × 6240 = 374 steps) causes quadratic ΔW ∝ LR² growth stall that collapses training. The cap brings warmup to ~30 steps so lora_B grows meaningfully within the first epoch and ES statistics become reliable. The LR decay phase is unaffected.
- **`--evorank_use_rslora`**: Defaults to `False` (standard LoRA scaling α/r). Set to `True` to enable rsLoRA (α/√r). rsLoRA can cause training instability on small datasets with high learning rates.

## Engineering Decisions (Paper–Code Gaps)

These implementation choices deviate from a naive reading of the paper and are not obvious from the algorithm description alone.

- **Optimizer state reset on rank change**: When a mutation is committed, AdamW `exp_avg` and `exp_avg_sq` are zeroed for the affected rank-1 slot (columns of lora_B, rows of lora_A). Without this, stale momentum from a previously-pruned slot corrupts the newly-activated one.
- **Gradient lower bound uses projected r×k matrix, not full d×k**: `_compute_gradient_rank1_direction()` runs power iteration on the implicit product of `∂L/∂B[:, active]` and `(A^T A + εI)^{-1} A^T`, avoiding materializing the full `(out, in)` gradient approximation — O(r(out+in)) per iteration, prevents OOM on large layers.
- **EMA initialization is direct assignment on first call**: On the first invocation of `update_statistics()`, `ema_u` and `ema_s` are set to the raw values directly (not `0.1 × value`). Cold-start fix to avoid the tracker spending many steps recovering from a zero-initialized state before the first rank decision.
- **Power iteration for rank-1 init runs 5 iterations**: `_power_iteration_rank1()` defaults to `num_iters=5`, sufficient for a reliable principal singular vector estimate at O(r(out+in)) per iter. Full SVD is never called on the weight gradient.
