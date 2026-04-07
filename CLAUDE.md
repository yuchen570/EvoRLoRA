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

## Architecture

### Core Files

| File | Role |
|------|------|
| `evo_rank_lora.py` | `EvoRankLoRALayer`: rank super-space (r_max), boolean active mask, gradient caching for ES |
| `rank_evolution_controller.py` | `RankEvolutionController`: expand/prune/reallocate mutations, EMA statistics, dynamic thresholds, cooldown |
| `train_integration.py` | `inject_evo_lora()` model injection; `train_evo_lora_step()` dual time-scale loop |
| `run_benchmark.py` | Main entry point (2558 lines): all CLI args, GLUE+NLG data loading, training loop, CSV export |
| `glue_metrics.py` | Task-specific metrics: Matthews (CoLA), F1 (MRPC/QQP), Pearson+Spearman mean (STS-B), Accuracy (rest) |
| `sora_inject.py` | `SoRALinear` + `SparseAdamW` proximal gradient optimizer |
| `toplora_inject.py` | `TopSingularValue` per-token singular value gating |
| `flatlora_inject.py` | Gaussian noise injection via hooks for flat minima regularization |
| `adalora_utils.py` | Orthogonal regularization loss + rank budget update helpers |

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
