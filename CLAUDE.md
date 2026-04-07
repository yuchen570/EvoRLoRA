# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EvoRLoRA is a research framework implementing **EvoRank-LoRA**, a dynamic rank adaptation method for LoRA based on Evolution Strategies. It benchmarks five PEFT methods (LoRA, AdaLoRA, EvoRank-LoRA, SoRA) on GLUE (NLU) and summarization (NLG) tasks.

## Environment Setup

```bash
conda create -y -p ./envs/evorank python=3.10
conda activate ./envs/evorank
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Running Experiments

**Smoke test (single GPU, ~2 min):**
```bash
python run_benchmark.py \
  --methods lora adalora evorank \
  --task_name sst2 \
  --model_name roberta-base \
  --max_train_steps 50 \
  --T_es 20 \
  --seed 42 \
  --output_dir artifacts \
  --export_csv results_smoke.csv
```

**Multi-GPU (DDP):**
```bash
torchrun --nproc_per_node=2 --master_port=29500 run_benchmark.py \
  --ddp --methods lora adalora evorank sora \
  --task_name sst2 --model_name roberta-base \
  --max_train_steps 20 --T_es 10 --seed 42 --output_dir artifacts
```

**Full benchmark scripts:**
```bash
bash scripts/fair_glue_deberta.sh          # GLUE 9 tasks, DeBERTa-v3-base, 5 seeds
bash scripts/fair_glue_deberta_rte.sh      # RTE (special: 50 epochs, batch=32)
bash scripts/fair_nlg_xsum.sh              # XSum, BART-large
bash scripts/fair_nlg_cnndailymail.sh      # CNN/DailyMail, BART-large
bash scripts/ablate_evorank_glue_deberta.sh  # Ablation studies
```

**Aggregate results:**
```bash
python scripts/summarize_evorank_ablation.py
python scripts/summarize_evorank_reallocation_efficiency.py
```

## Architecture

### Core EvoRank-LoRA

**`evo_rank_lora.py` — `EvoRankLoRALayer`**
Rank super-space parameterization: maintains `lora_A` (r_max × in) and `lora_B` (out × r_max) with a binary `active_mask`. Forward pass computes ΔW = Σ m_i · b_i · a_i^T. Key methods: `activate_component()`, `deactivate_component()`, `compute_demand_score()`, `compute_prune_scores()`.

**`rank_evolution_controller.py` — `RankEvolutionController`**
Manages ES trials and three mutation types (`ExpandMutation`, `PruneMutation`, `ReallocateMutation`). Uses EMA-smoothed gradient statistics, quantile-based dynamic thresholds, persistent counters with cooldown, and complexity regularization (−λ_c × C(z)). In DDP mode, trial rewards are synchronized via `all_reduce`.

**`train_integration.py`**
- `inject_evo_lora()`: Replaces target `nn.Linear` layers with `EvoRankLoRAWrapper` (frozen base + `EvoRankLoRALayer`)
- `train_evo_lora_step()`: Dual-timescale loop — caches gradients, runs ES trial evaluation on a mini validation set, applies/evaluates/undoes mutations, commits elitist selection

### Comparison Methods

| File | Method | Key mechanism |
|------|--------|---------------|
| `sora_inject.py` | SoRA | Learnable sparse gate + `SparseAdamW` proximal optimizer with L1 soft-thresholding |
| `adalora_utils.py` | AdaLoRA | Wraps HuggingFace PEFT AdaLoRA with orthogonal regularization loss |

### Evaluation

**`glue_metrics.py`**: Task-specific primary metrics (MCC for CoLA, F1 for MRPC/QQP, Pearson+Spearman avg for STS-B, accuracy for others). Used by `run_benchmark.py` for best-checkpoint tracking.

**`run_benchmark.py`** (~3000 lines): Unified entry point. Handles data loading, model init, optimizer/scheduler setup, training loop, validation, TensorBoard/W&B/CSV logging, and checkpointing for all 5 methods.

### Output Layout

```
artifacts/<task>_<backbone>_<method>/
├── metrics.jsonl          # per-step metrics
├── final/                 # saved model + tokenizer
└── checkpoint_epoch_*.pt  # optional per-epoch checkpoints
results_*.csv              # aggregated mean/std across seeds
runs/                      # TensorBoard event files
```

## Key EvoRank Hyperparameters

| Flag | Default | Meaning |
|------|---------|---------|
| `--target_rank` | — | Initial active rank (r_init) |
| `--evorank_r_max` | 16 | Rank super-space ceiling R_max |
| `--T_es` | 200 | ES population size (steps between ES evaluations) |
| `--lambda_c` | 0.001 | Complexity regularization weight |
| `--complexity_mode` | `rank_sum` | `rank_sum` or `size_aware` |
| `--expand_init_mode` | `zero` | `zero` (cold start) or `gradient` (Prop 3.2) |
| `--lambda_pop` | 16 | Population subsampling limit |
| `--evo_max_reallocate_candidates` | 8 | Cross-layer reallocation candidate limit |
