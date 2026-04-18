# Windows & Linux 单卡 NLU（GLUE）Python 完整指令

与 [`fair_glue_deberta_common.sh`](fair_glue_deberta_common.sh) 及各 `fair_glue_deberta_<task>.sh` **超参与 batch 对齐**；在**仓库根目录**执行（需存在 `run_benchmark.py`）。此处为**单进程前台**运行：不用 `torchrun`、不设 `master_port`、不写后台日志文件。

## 前提

- 工作目录示例：`cd D:\EvoRLoRA`
- 建议创建目录：`New-Item -ItemType Directory -Force -Path runs,artifacts`
- 可选固定 GPU：`$env:CUDA_VISIBLE_DEVICES="0"`
- **batch**：`--batch_size 32`（与公平脚本中「1 卡 × 每卡 32」的全局 batch 一致时，AdaLoRA 的 `tinit/tfinal/delta_t` 与下表数值一致，无需缩放）

## 共用参数说明

除下表中 **各任务超参** 外，与 `fair_glue_deberta_common.sh` 中 `run_task` 一致；与 bash 的差异仅为**不传入 `--ddp`**（单卡单进程，非 DistributedDataParallel）。

- 方法：`lora adalora evorank sora toplora flatlora pissa`
- 协议：`controlled_fair`，`protocol_dropout=0.05`
- 模型：`microsoft/deberta-v3-base`，`target_rank=8`，`module_preset=default`，`flatlora_rho=0.05`
- 训练：`warmup_ratio=0.06`，`max_grad_norm=1.0`，`batch_size=32`
- EvoRank 等：`lambda_c=0.0`，`expand_init_mode=gradient`，`evo_compensation_mode=B`，`mini_val_k=16`，`evo_alpha_u=1.5`，`evo_p_g=0.75`，`evo_p_p=0.03`，`evo_H_p=6`，`evo_cooldown_steps=5`，`evo_max_reallocate_candidates=16`，`verify_n_samples=0`
- 种子：`0 21 42 81 100`
- SORA：`--sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4`

## 指令模板 (PowerShell & Bash)

将 `<TASK>` 及下表中的占位符替换为实际值：

#### PowerShell (Windows)

```powershell
$env:CUDA_VISIBLE_DEVICES="0"   # 可选

python run_benchmark.py `
  --methods lora adalora evorank sora toplora flatlora pissa `
  --comparison_protocol controlled_fair --protocol_dropout 0.05 `
  --module_preset default --flatlora_rho 0.05 `
  --task_list <TASK> `
  --model_list microsoft/deberta-v3-base `
  --target_rank 8 `
  --lora_alpha <ALPHA> `
  --epochs <EPOCHS> `
  --batch_size 32 `
  --max_length <MAX_LEN> `
  --lr <LR> `
  --warmup_ratio 0.06 `
  --weight_decay <WD> `
  --max_grad_norm 1.0 `
  --adalora_tinit <TINIT> --adalora_tfinal <TFINAL> --adalora_delta_t <DELTA_T> --adalora_orth_reg_weight <ORTH> `
  --sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4 `
  --lambda_c 0.0 `
  --expand_init_mode gradient --evo_compensation_mode B `
  --mini_val_k 16 --evo_alpha_u 1.5 --evo_p_g 0.75 --evo_p_p 0.03 --evo_H_p 6 --evo_cooldown_steps 5 --evo_max_reallocate_candidates 16 `
  --verify_n_samples 0 `
  --seed_list 0 21 42 81 100 `
  --log_dir runs/fair_glue_deberta_<TASK> `
  --output_dir artifacts `
  --export_csv results_fair_glue_deberta_<TASK>.csv
```

#### Bash (Linux)

```bash
# 可选
CUDA_VISIBLE_DEVICES="0" python run_benchmark.py \
  --methods lora adalora evorank sora toplora flatlora pissa \
  --comparison_protocol controlled_fair --protocol_dropout 0.05 \
  --module_preset default --flatlora_rho 0.05 \
  --task_list <TASK> \
  --model_list microsoft/deberta-v3-base \
  --target_rank 8 \
  --lora_alpha <ALPHA> \
  --epochs <EPOCHS> \
  --batch_size 32 \
  --max_length <MAX_LEN> \
  --lr <LR> \
  --warmup_ratio 0.06 \
  --weight_decay <WD> \
  --max_grad_norm 1.0 \
  --adalora_tinit <TINIT> --adalora_tfinal <TFINAL> --adalora_delta_t <DELTA_T> --adalora_orth_reg_weight <ORTH> \
  --sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4 \
  --lambda_c 0.0 \
  --expand_init_mode gradient --evo_compensation_mode B \
  --mini_val_k 16 --evo_alpha_u 1.5 --evo_p_g 0.75 --evo_p_p 0.03 --evo_H_p 6 --evo_cooldown_steps 5 --evo_max_reallocate_candidates 16 \
  --verify_n_samples 0 \
  --seed_list 0 21 42 81 100 \
  --log_dir runs/fair_glue_deberta_<TASK> \
  --output_dir artifacts \
  --export_csv results_fair_glue_deberta_<TASK>.csv
```

## 各 GLUE 任务：`task_list` 与超参

（bash 中的 `master_port` 仅用于 `torchrun` 多进程，此处直接 `python` 运行不需要。）

| 任务 | lr | epochs | max_length | lora_alpha | weight_decay | adalora_tinit | adalora_tfinal | adalora_delta_t | adalora_orth_reg_weight |
|------|-----|--------|------------|------------|----------------|---------------|----------------|-----------------|-------------------------|
| cola | 8e-4 | 25 | 64 | 32 | 0.0 | 800 | 3500 | 10 | 0.1 |
| mnli | 5e-4 | 7 | 256 | 16 | 0.0 | 8000 | 50000 | 100 | 0.1 |
| mrpc | 1e-3 | 30 | 320 | 32 | 0.01 | 600 | 1800 | 1 | 0.1 |
| qqp | 8e-4 | 5 | 320 | 16 | 0.01 | 8000 | 25000 | 100 | 0.1 |
| qnli | 5e-4 | 5 | 512 | 32 | 0.01 | 2000 | 8000 | 100 | 0.1 |
| rte | 1.2e-3 | 50 | 320 | 32 | 0.01 | 600 | 1800 | 1 | 0.3 |
| sst2 | 8e-4 | 24 | 128 | 16 | 0.01 | 6000 | 22000 | 100 | 0.1 |
| stsb | 2.2e-3 | 25 | 128 | 32 | 0.1 | 800 | 2000 | 10 | 0.3 |

## 各任务完整指令（无占位符，与 `fair_glue_deberta_<task>.sh` 对齐）

以下每条与 [`fair_glue_deberta_common.sh`](fair_glue_deberta_common.sh) 中 `run_task` 传入的 **lr / epochs / max_length / lora_alpha / weight_decay / adalora_\*** 一致（单进程下未使用 bash 的 AdaLoRA 步数缩放；默认全局 batch 为 32 时与 ref 值相同）。共用段与上文「共用参数说明」相同。

### CoLA（[`fair_glue_deberta_cola.sh`](fair_glue_deberta_cola.sh)）

#### PowerShell (Windows)

```powershell
$env:CUDA_VISIBLE_DEVICES="0"

python run_benchmark.py `
  --methods lora adalora evorank sora toplora flatlora pissa `
  --comparison_protocol controlled_fair --protocol_dropout 0.05 `
  --module_preset default --flatlora_rho 0.05 `
  --task_list cola `
  --model_list microsoft/deberta-v3-base `
  --target_rank 8 `
  --lora_alpha 32 `
  --epochs 25 `
  --batch_size 32 `
  --max_length 64 `
  --lr 8e-4 `
  --warmup_ratio 0.06 `
  --weight_decay 0.0 `
  --max_grad_norm 1.0 `
  --adalora_tinit 800 --adalora_tfinal 3500 --adalora_delta_t 10 --adalora_orth_reg_weight 0.1 `
  --sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4 `
  --lambda_c 0.0 `
  --expand_init_mode gradient --evo_compensation_mode B `
  --mini_val_k 16 --evo_alpha_u 1.5 --evo_p_g 0.75 --evo_p_p 0.03 --evo_H_p 6 --evo_cooldown_steps 5 --evo_max_reallocate_candidates 16 `
  --verify_n_samples 0 `
  --seed_list 0 21 42 81 100 `
  --log_dir runs/fair_glue_deberta_cola `
  --output_dir artifacts `
  --export_csv results_fair_glue_deberta_cola.csv
```

#### Bash (Linux)

```bash
CUDA_VISIBLE_DEVICES="0" python run_benchmark.py \
  --methods lora adalora evorank sora toplora flatlora pissa \
  --comparison_protocol controlled_fair --protocol_dropout 0.05 \
  --module_preset default --flatlora_rho 0.05 \
  --task_list cola \
  --model_list microsoft/deberta-v3-base \
  --target_rank 8 \
  --lora_alpha 32 \
  --epochs 25 \
  --batch_size 32 \
  --dataloader_num_workers 8 \
  --pin_memory \
  --max_length 64 \
  --lr 8e-4 \
  --warmup_ratio 0.06 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --adalora_tinit 800 --adalora_tfinal 3500 --adalora_delta_t 10 --adalora_orth_reg_weight 0.1 \
  --sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4 \
  --lambda_c 0.0 \
  --expand_init_mode gradient --evo_compensation_mode B \
  --mini_val_k 16 --evo_alpha_u 1.5 --evo_p_g 0.75 --evo_p_p 0.03 --evo_H_p 6 --evo_cooldown_steps 5 --evo_max_reallocate_candidates 16 \
  --verify_n_samples 0 \
  --seed_list 0 21 42 81 100 \
  --log_dir runs/fair_glue_deberta_cola \
  --output_dir artifacts \
  --export_csv results_fair_glue_deberta_cola.csv
```

### MNLI（[`fair_glue_deberta_mnli.sh`](fair_glue_deberta_mnli.sh)）

#### PowerShell (Windows)

```powershell
$env:CUDA_VISIBLE_DEVICES="0"

python run_benchmark.py `
  --methods lora adalora evorank sora toplora flatlora pissa `
  --comparison_protocol controlled_fair --protocol_dropout 0.05 `
  --module_preset default --flatlora_rho 0.05 `
  --task_list mnli `
  --model_list microsoft/deberta-v3-base `
  --target_rank 8 `
  --lora_alpha 16 `
  --epochs 7 `
  --batch_size 32 `
  --max_length 256 `
  --lr 5e-4 `
  --warmup_ratio 0.06 `
  --weight_decay 0.0 `
  --max_grad_norm 1.0 `
  --adalora_tinit 8000 --adalora_tfinal 50000 --adalora_delta_t 100 --adalora_orth_reg_weight 0.1 `
  --sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4 `
  --lambda_c 0.0 `
  --expand_init_mode gradient --evo_compensation_mode B `
  --mini_val_k 16 --evo_alpha_u 1.5 --evo_p_g 0.75 --evo_p_p 0.03 --evo_H_p 6 --evo_cooldown_steps 5 --evo_max_reallocate_candidates 16 `
  --verify_n_samples 0 `
  --seed_list 0 21 42 81 100 `
  --log_dir runs/fair_glue_deberta_mnli `
  --output_dir artifacts `
  --export_csv results_fair_glue_deberta_mnli.csv
```

#### Bash (Linux)

```bash
CUDA_VISIBLE_DEVICES="0" python run_benchmark.py \
  --methods lora adalora evorank sora toplora flatlora pissa \
  --comparison_protocol controlled_fair --protocol_dropout 0.05 \
  --module_preset default --flatlora_rho 0.05 \
  --task_list mnli \
  --model_list microsoft/deberta-v3-base \
  --target_rank 8 \
  --lora_alpha 16 \
  --epochs 7 \
  --batch_size 32 \
  --max_length 256 \
  --lr 5e-4 \
  --warmup_ratio 0.06 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --adalora_tinit 8000 --adalora_tfinal 50000 --adalora_delta_t 100 --adalora_orth_reg_weight 0.1 \
  --sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4 \
  --lambda_c 0.0 \
  --expand_init_mode gradient --evo_compensation_mode B \
  --mini_val_k 16 --evo_alpha_u 1.5 --evo_p_g 0.75 --evo_p_p 0.03 --evo_H_p 6 --evo_cooldown_steps 5 --evo_max_reallocate_candidates 16 \
  --verify_n_samples 0 \
  --seed_list 0 21 42 81 100 \
  --log_dir runs/fair_glue_deberta_mnli \
  --output_dir artifacts \
  --export_csv results_fair_glue_deberta_mnli.csv
```

### MRPC（[`fair_glue_deberta_mrpc.sh`](fair_glue_deberta_mrpc.sh)）

#### PowerShell (Windows)

```powershell
$env:CUDA_VISIBLE_DEVICES="0"

python run_benchmark.py `
  --methods lora adalora evorank sora toplora flatlora pissa `
  --comparison_protocol controlled_fair --protocol_dropout 0.05 `
  --module_preset default --flatlora_rho 0.05 `
  --task_list mrpc `
  --model_list microsoft/deberta-v3-base `
  --target_rank 8 `
  --lora_alpha 32 `
  --epochs 30 `
  --batch_size 32 `
  --max_length 320 `
  --lr 1e-3 `
  --warmup_ratio 0.06 `
  --weight_decay 0.01 `
  --max_grad_norm 1.0 `
  --adalora_tinit 600 --adalora_tfinal 1800 --adalora_delta_t 1 --adalora_orth_reg_weight 0.1 `
  --sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4 `
  --lambda_c 0.0 `
  --expand_init_mode gradient --evo_compensation_mode B `
  --mini_val_k 16 --evo_alpha_u 1.5 --evo_p_g 0.75 --evo_p_p 0.03 --evo_H_p 6 --evo_cooldown_steps 5 --evo_max_reallocate_candidates 16 `
  --verify_n_samples 0 `
  --seed_list 0 21 42 81 100 `
  --log_dir runs/fair_glue_deberta_mrpc `
  --output_dir artifacts `
  --export_csv results_fair_glue_deberta_mrpc.csv
```

#### Bash (Linux)

```bash
CUDA_VISIBLE_DEVICES="0" python run_benchmark.py \
  --methods lora adalora evorank sora toplora flatlora pissa \
  --comparison_protocol controlled_fair --protocol_dropout 0.05 \
  --module_preset default --flatlora_rho 0.05 \
  --task_list mrpc \
  --model_list microsoft/deberta-v3-base \
  --target_rank 8 \
  --lora_alpha 32 \
  --epochs 30 \
  --batch_size 32 \
  --max_length 320 \
  --lr 1e-3 \
  --warmup_ratio 0.06 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --adalora_tinit 600 --adalora_tfinal 1800 --adalora_delta_t 1 --adalora_orth_reg_weight 0.1 \
  --sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4 \
  --lambda_c 0.0 \
  --expand_init_mode gradient --evo_compensation_mode B \
  --mini_val_k 16 --evo_alpha_u 1.5 --evo_p_g 0.75 --evo_p_p 0.03 --evo_H_p 6 --evo_cooldown_steps 5 --evo_max_reallocate_candidates 16 \
  --verify_n_samples 0 \
  --seed_list 0 21 42 81 100 \
  --log_dir runs/fair_glue_deberta_mrpc \
  --output_dir artifacts \
  --export_csv results_fair_glue_deberta_mrpc.csv
```

### QQP（[`fair_glue_deberta_qqp.sh`](fair_glue_deberta_qqp.sh)）

#### PowerShell (Windows)

```powershell
$env:CUDA_VISIBLE_DEVICES="0"

python run_benchmark.py `
  --methods lora adalora evorank sora toplora flatlora pissa `
  --comparison_protocol controlled_fair --protocol_dropout 0.05 `
  --module_preset default --flatlora_rho 0.05 `
  --task_list qqp `
  --model_list microsoft/deberta-v3-base `
  --target_rank 8 `
  --lora_alpha 16 `
  --epochs 5 `
  --batch_size 32 `
  --max_length 320 `
  --lr 8e-4 `
  --warmup_ratio 0.06 `
  --weight_decay 0.01 `
  --max_grad_norm 1.0 `
  --adalora_tinit 8000 --adalora_tfinal 25000 --adalora_delta_t 100 --adalora_orth_reg_weight 0.1 `
  --sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4 `
  --lambda_c 0.0 `
  --expand_init_mode gradient --evo_compensation_mode B `
  --mini_val_k 16 --evo_alpha_u 1.5 --evo_p_g 0.75 --evo_p_p 0.03 --evo_H_p 6 --evo_cooldown_steps 5 --evo_max_reallocate_candidates 16 `
  --verify_n_samples 0 `
  --seed_list 0 21 42 81 100 `
  --log_dir runs/fair_glue_deberta_qqp `
  --output_dir artifacts `
  --export_csv results_fair_glue_deberta_qqp.csv
```

#### Bash (Linux)

```bash
CUDA_VISIBLE_DEVICES="0" python run_benchmark.py \
  --methods lora adalora evorank sora toplora flatlora pissa \
  --comparison_protocol controlled_fair --protocol_dropout 0.05 \
  --module_preset default --flatlora_rho 0.05 \
  --task_list qqp \
  --model_list microsoft/deberta-v3-base \
  --target_rank 8 \
  --lora_alpha 16 \
  --epochs 5 \
  --batch_size 32 \
  --max_length 320 \
  --lr 8e-4 \
  --warmup_ratio 0.06 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --adalora_tinit 8000 --adalora_tfinal 25000 --adalora_delta_t 100 --adalora_orth_reg_weight 0.1 \
  --sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4 \
  --lambda_c 0.0 \
  --expand_init_mode gradient --evo_compensation_mode B \
  --mini_val_k 16 --evo_alpha_u 1.5 --evo_p_g 0.75 --evo_p_p 0.03 --evo_H_p 6 --evo_cooldown_steps 5 --evo_max_reallocate_candidates 16 \
  --verify_n_samples 0 \
  --seed_list 0 21 42 81 100 \
  --log_dir runs/fair_glue_deberta_qqp \
  --output_dir artifacts \
  --export_csv results_fair_glue_deberta_qqp.csv
```

### QNLI（[`fair_glue_deberta_qnli.sh`](fair_glue_deberta_qnli.sh)）

#### PowerShell (Windows)

```powershell
$env:CUDA_VISIBLE_DEVICES="0"

python run_benchmark.py `
  --methods lora adalora evorank sora toplora flatlora pissa `
  --comparison_protocol controlled_fair --protocol_dropout 0.05 `
  --module_preset default --flatlora_rho 0.05 `
  --task_list qnli `
  --model_list microsoft/deberta-v3-base `
  --target_rank 8 `
  --lora_alpha 32 `
  --epochs 5 `
  --batch_size 32 `
  --max_length 512 `
  --lr 5e-4 `
  --warmup_ratio 0.06 `
  --weight_decay 0.01 `
  --max_grad_norm 1.0 `
  --adalora_tinit 2000 --adalora_tfinal 8000 --adalora_delta_t 100 --adalora_orth_reg_weight 0.1 `
  --sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4 `
  --lambda_c 0.0 `
  --expand_init_mode gradient --evo_compensation_mode B `
  --mini_val_k 16 --evo_alpha_u 1.5 --evo_p_g 0.75 --evo_p_p 0.03 --evo_H_p 6 --evo_cooldown_steps 5 --evo_max_reallocate_candidates 16 `
  --verify_n_samples 0 `
  --seed_list 0 21 42 81 100 `
  --log_dir runs/fair_glue_deberta_qnli `
  --output_dir artifacts `
  --export_csv results_fair_glue_deberta_qnli.csv
```

#### Bash (Linux)

```bash
CUDA_VISIBLE_DEVICES="0" python run_benchmark.py \
  --methods lora adalora evorank sora toplora flatlora pissa \
  --comparison_protocol controlled_fair --protocol_dropout 0.05 \
  --module_preset default --flatlora_rho 0.05 \
  --task_list qnli \
  --model_list microsoft/deberta-v3-base \
  --target_rank 8 \
  --lora_alpha 32 \
  --epochs 5 \
  --batch_size 32 \
  --max_length 512 \
  --lr 5e-4 \
  --warmup_ratio 0.06 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --adalora_tinit 2000 --adalora_tfinal 8000 --adalora_delta_t 100 --adalora_orth_reg_weight 0.1 \
  --sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4 \
  --lambda_c 0.0 \
  --expand_init_mode gradient --evo_compensation_mode B \
  --mini_val_k 16 --evo_alpha_u 1.5 --evo_p_g 0.75 --evo_p_p 0.03 --evo_H_p 6 --evo_cooldown_steps 5 --evo_max_reallocate_candidates 16 \
  --verify_n_samples 0 \
  --seed_list 0 21 42 81 100 \
  --log_dir runs/fair_glue_deberta_qnli \
  --output_dir artifacts \
  --export_csv results_fair_glue_deberta_qnli.csv
```

### RTE（[`fair_glue_deberta_rte.sh`](fair_glue_deberta_rte.sh)）

#### PowerShell (Windows)

```powershell
$env:CUDA_VISIBLE_DEVICES="0"

python run_benchmark.py `
  --methods lora adalora evorank sora toplora flatlora pissa `
  --comparison_protocol controlled_fair --protocol_dropout 0.05 `
  --module_preset default --flatlora_rho 0.05 `
  --task_list rte `
  --model_list microsoft/deberta-v3-base `
  --target_rank 8 `
  --lora_alpha 32 `
  --epochs 50 `
  --batch_size 32 `
  --max_length 320 `
  --lr 1.2e-3 `
  --warmup_ratio 0.06 `
  --weight_decay 0.01 `
  --max_grad_norm 1.0 `
  --adalora_tinit 600 --adalora_tfinal 1800 --adalora_delta_t 1 --adalora_orth_reg_weight 0.3 `
  --sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4 `
  --lambda_c 0.0 `
  --expand_init_mode gradient --evo_compensation_mode B `
  --mini_val_k 16 --evo_alpha_u 1.5 --evo_p_g 0.75 --evo_p_p 0.03 --evo_H_p 6 --evo_cooldown_steps 5 --evo_max_reallocate_candidates 16 `
  --verify_n_samples 0 `
  --seed_list 0 21 42 81 100 `
  --log_dir runs/fair_glue_deberta_rte `
  --output_dir artifacts `
  --export_csv results_fair_glue_deberta_rte.csv
```

#### Bash (Linux)

```bash
CUDA_VISIBLE_DEVICES="0" python run_benchmark.py \
  --methods lora adalora evorank sora toplora flatlora pissa \
  --comparison_protocol controlled_fair --protocol_dropout 0.05 \
  --module_preset default --flatlora_rho 0.05 \
  --task_list rte \
  --model_list microsoft/deberta-v3-base \
  --target_rank 8 \
  --lora_alpha 32 \
  --epochs 50 \
  --batch_size 32 \
  --max_length 320 \
  --lr 1.2e-3 \
  --warmup_ratio 0.06 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --adalora_tinit 600 --adalora_tfinal 1800 --adalora_delta_t 1 --adalora_orth_reg_weight 0.3 \
  --sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4 \
  --lambda_c 0.0 \
  --expand_init_mode gradient --evo_compensation_mode B \
  --mini_val_k 16 --evo_alpha_u 1.5 --evo_p_g 0.75 --evo_p_p 0.03 --evo_H_p 6 --evo_cooldown_steps 5 --evo_max_reallocate_candidates 16 \
  --verify_n_samples 0 \
  --seed_list 0 21 42 81 100 \
  --log_dir runs/fair_glue_deberta_rte \
  --output_dir artifacts \
  --export_csv results_fair_glue_deberta_rte.csv
```

### SST-2（[`fair_glue_deberta_sst2.sh`](fair_glue_deberta_sst2.sh)）

#### PowerShell (Windows)

```powershell
$env:CUDA_VISIBLE_DEVICES="0"

python run_benchmark.py `
  --methods lora adalora evorank sora toplora flatlora pissa `
  --comparison_protocol controlled_fair --protocol_dropout 0.05 `
  --module_preset default --flatlora_rho 0.05 `
  --task_list sst2 `
  --model_list microsoft/deberta-v3-base `
  --target_rank 8 `
  --lora_alpha 16 `
  --epochs 24 `
  --batch_size 32 `
  --max_length 128 `
  --lr 8e-4 `
  --warmup_ratio 0.06 `
  --weight_decay 0.01 `
  --max_grad_norm 1.0 `
  --adalora_tinit 6000 --adalora_tfinal 22000 --adalora_delta_t 100 --adalora_orth_reg_weight 0.1 `
  --sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4 `
  --lambda_c 0.0 `
  --expand_init_mode gradient --evo_compensation_mode B `
  --mini_val_k 16 --evo_alpha_u 1.5 --evo_p_g 0.75 --evo_p_p 0.03 --evo_H_p 6 --evo_cooldown_steps 5 --evo_max_reallocate_candidates 16 `
  --verify_n_samples 0 `
  --seed_list 0 21 42 81 100 `
  --log_dir runs/fair_glue_deberta_sst2 `
  --output_dir artifacts `
  --export_csv results_fair_glue_deberta_sst2.csv
```

#### Bash (Linux)

```bash
CUDA_VISIBLE_DEVICES="0" python run_benchmark.py \
  --methods lora adalora evorank sora toplora flatlora pissa \
  --comparison_protocol controlled_fair --protocol_dropout 0.05 \
  --module_preset default --flatlora_rho 0.05 \
  --task_list sst2 \
  --model_list microsoft/deberta-v3-base \
  --target_rank 8 \
  --lora_alpha 16 \
  --epochs 24 \
  --batch_size 32 \
  --max_length 128 \
  --lr 8e-4 \
  --warmup_ratio 0.06 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --adalora_tinit 6000 --adalora_tfinal 22000 --adalora_delta_t 100 --adalora_orth_reg_weight 0.1 \
  --sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4 \
  --lambda_c 0.0 \
  --expand_init_mode gradient --evo_compensation_mode B \
  --mini_val_k 16 --evo_alpha_u 1.5 --evo_p_g 0.75 --evo_p_p 0.03 --evo_H_p 6 --evo_cooldown_steps 5 --evo_max_reallocate_candidates 16 \
  --verify_n_samples 0 \
  --seed_list 0 21 42 81 100 \
  --log_dir runs/fair_glue_deberta_sst2 \
  --output_dir artifacts \
  --export_csv results_fair_glue_deberta_sst2.csv
```

### STS-B（[`fair_glue_deberta_stsb.sh`](fair_glue_deberta_stsb.sh)）

#### PowerShell (Windows)

```powershell
$env:CUDA_VISIBLE_DEVICES="0"

python run_benchmark.py `
  --methods lora adalora evorank sora toplora flatlora pissa `
  --comparison_protocol controlled_fair --protocol_dropout 0.05 `
  --module_preset default --flatlora_rho 0.05 `
  --task_list stsb `
  --model_list microsoft/deberta-v3-base `
  --target_rank 8 `
  --lora_alpha 32 `
  --epochs 25 `
  --batch_size 32 `
  --max_length 128 `
  --lr 2.2e-3 `
  --warmup_ratio 0.06 `
  --weight_decay 0.1 `
  --max_grad_norm 1.0 `
  --adalora_tinit 800 --adalora_tfinal 2000 --adalora_delta_t 10 --adalora_orth_reg_weight 0.3 `
  --sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4 `
  --lambda_c 0.0 `
  --expand_init_mode gradient --evo_compensation_mode B `
  --mini_val_k 16 --evo_alpha_u 1.5 --evo_p_g 0.75 --evo_p_p 0.03 --evo_H_p 6 --evo_cooldown_steps 5 --evo_max_reallocate_candidates 16 `
  --verify_n_samples 0 `
  --seed_list 0 21 42 81 100 `
  --log_dir runs/fair_glue_deberta_stsb `
  --output_dir artifacts `
  --export_csv results_fair_glue_deberta_stsb.csv
```

#### Bash (Linux)

```bash
CUDA_VISIBLE_DEVICES="0" python run_benchmark.py \
  --methods lora adalora evorank sora toplora flatlora pissa \
  --comparison_protocol controlled_fair --protocol_dropout 0.05 \
  --module_preset default --flatlora_rho 0.05 \
  --task_list stsb \
  --model_list microsoft/deberta-v3-base \
  --target_rank 8 \
  --lora_alpha 32 \
  --epochs 25 \
  --batch_size 32 \
  --max_length 128 \
  --lr 2.2e-3 \
  --warmup_ratio 0.06 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --adalora_tinit 800 --adalora_tfinal 2000 --adalora_delta_t 10 --adalora_orth_reg_weight 0.3 \
  --sora_sparse_lambda 10 --sora_sparse_lambda_2 3e-4 \
  --lambda_c 0.0 \
  --expand_init_mode gradient --evo_compensation_mode B \
  --mini_val_k 16 --evo_alpha_u 1.5 --evo_p_g 0.75 --evo_p_p 0.03 --evo_H_p 6 --evo_cooldown_steps 5 --evo_max_reallocate_candidates 16 \
  --verify_n_samples 0 \
  --seed_list 0 21 42 81 100 \
  --log_dir runs/fair_glue_deberta_stsb \
  --output_dir artifacts \
  --export_csv results_fair_glue_deberta_stsb.csv
```

## 其它说明

- [`ablate_evorank_glue_deberta.sh`](ablate_evorank_glue_deberta.sh) 使用多卡 `torchrun`；若改为本机单进程，需自行统一 **全局 batch** 与步数含义后再对比消融结果。
- 汇总表格可尝试 [`generate_glue_table.py`](generate_glue_table.py)，使用前请确认其期望的 CSV 文件名与 `results_fair_glue_deberta_*.csv` 是否匹配。
