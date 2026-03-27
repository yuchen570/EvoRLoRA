# EvoRank-LoRA

本仓库实现了一个可演化的 LoRA 框架（EvoRank-LoRA），核心包括：

- `EvoRankLoRALayer`：最大秩超空间 + 掩码激活 + 在线扩张/修剪
- `RankEvolutionController`：EMA 统计、动态阈值、计数器与冷却、Mutation 生成与提交
- `train_integration.py`：模型注入（`inject_evo_lora`）与双时间尺度训练步（`train_evo_lora_step`）
- `run_benchmark.py`：GLUE 基准脚本（LoRA/AdaLoRA/EvoRank），支持日志与 CSV 导出

---

## 近期更新（与论文实现一致性相关）

当前代码已支持以下关键机制：

- 动态阈值（分位数）+ EMA 平滑
- 持久计数器（`H_g` / `H_p`）+ cooldown
- Rolling Candidacy（落选候选计数保留）
- DDP Trial 奖励同步（`all_reduce`）
- no-op 候选参与 elitist 选择
- Reward 复杂度正则：`-lambda_c * C(z)`，支持：
  - `rank_sum`
  - `size_aware`
- ES population 子采样：
  - `--lambda_pop`
  - `--population_strategy all|random`
- 学术协议批量运行：
  - `--task_list`
  - `--model_list`
  - `--export_csv`

---

## 环境准备（Ubuntu + A800 + CUDA 13 驱动）

你的机器信息（示例）：

- OS: Ubuntu Linux x86_64
- GPU: NVIDIA A800 80GB
- Driver: 580.126.09
- Reported CUDA: 13.0

### 重要说明（PyTorch 与 CUDA 13）

很多 PyTorch 预编译 wheel 目前并不直接以 `cu130` 发布。推荐做法是：

- 使用**较新 NVIDIA 驱动**（你已满足）
- 安装 PyTorch 官方提供的 `cu121` 或 `cu124` wheel（通常可在 CUDA 13 驱动上正常运行）

即：驱动向后兼容运行时，不必强行本地安装 CUDA Toolkit 13 才能跑 PyTorch。

---

## 安装步骤

### 1) 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) 先安装 PyTorch（单独安装，避免 requirements 与 CUDA 轮子冲突）

二选一（推荐先尝试 cu124）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3) 安装其余依赖

```bash
pip install -r requirements.txt
```

---

## 快速开始
### 1) 冒烟测试（推荐先跑）

```bash
python run_benchmark.py \
  --methods lora adalora evorank \
  --task_name sst2 \
  --model_name roberta-base \
  --max_train_steps 50 \
  --T_es 20 \
  --warmup_ratio 0.1 \
  --seed 42 \
  --log_dir runs/smoke \
  --export_csv results_smoke.csv
```

如果你要和你当前终端里的命令保持一致（例如加入 `--use_wandb`），可以在上面基础上补充：
`--use_wandb --wandb_project <your_project>`

### 2) 主结果模板（多任务）

```bash
python run_benchmark.py \
  --methods lora adalora evorank \
  --task_list sst2 qnli rte mnli \
  --model_list roberta-base \
  --target_rank 8 \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --T_es 200 \
  --mini_val_k 8 \
  --adalora_delta_t 200 \
  --lambda_c 0.0 \
  --complexity_mode rank_sum \
  --population_strategy all \
  --seed 42 \
  --log_dir runs/main_results \
  --export_csv results_main.csv
```

### 3) 消融模板（EvoRank）

```bash
python run_benchmark.py \
  --methods evorank \
  --task_name sst2 \
  --model_name roberta-base \
  --target_rank 8 \
  --epochs 3 \
  --lr 2e-5 \
  --T_es 200 \
  --warmup_ratio 0.1 \
  --lambda_c 0.001 \
  --complexity_mode rank_sum \
  --lambda_pop 16 \
  --population_strategy all \
  --seed 42 \
  --log_dir runs/ablation/full \
  --export_csv results_ablation_full.csv
```

### 4) 效率模板

```bash
python run_benchmark.py \
  --methods lora adalora evorank \
  --task_name sst2 \
  --model_name roberta-base \
  --target_rank 8 \
  --batch_size 16 \
  --max_train_steps 500 \
  --lr 2e-5 \
  --warmup_ratio 0.1 \
  --T_es 200 \
  --mini_val_k 8 \
  --adalora_delta_t 200 \
  --lambda_c 0.001 \
  --complexity_mode rank_sum \
  --lambda_pop 16 \
  --population_strategy all \
  --seed 42 \
  --log_dir runs/efficiency \
  --export_csv results_efficiency.csv
```

---

## 日志与结果

- TensorBoard：
  - 默认目录：`runs/...`
  - 启动：`tensorboard --logdir runs`
- W&B（可选）：
  - 命令增加 `--use_wandb --wandb_project <project_name>`
- CSV 导出：
  - `--export_csv <file.csv>`
  - 默认字段：`task/backbone/method/trainable_params/best_val_accuracy/peak_memory_mb/avg_active_rank/total_train_time_sec`

---

## 常见问题

### 1) AdaLoRA 报错：`total_step is None`

确保 `peft_factory` 初始化 AdaLoRA 时传入了 `total_step`。本仓库已做兼容处理（自动按计划步数注入）。

### 2) EvoRank 精度接近随机

检查分类头是否解冻。HF 的 LoRA/AdaLoRA 在 `SEQ_CLS` 任务通常会处理任务头；EvoRank 手动注入需要显式解冻 `classifier`/`score` 参数。本仓库已处理。

### 3) 报设备不一致（CPU/CUDA）

典型报错：`Expected all tensors to be on the same device`。本仓库在 `train_evo_lora_step` 中已加入 controller 状态张量对齐。

---

## 主要脚本

- `evo_rank_lora.py`：可演化 LoRA 层
- `rank_evolution_controller.py`：演化控制器与 Mutation 体系
- `train_integration.py`：注入与双时间尺度训练
- `run_benchmark.py`：主实验入口
