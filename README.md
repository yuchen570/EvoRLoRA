# EvoRank-LoRA

本仓库实现了一个可演化的 LoRA 框架（EvoRank-LoRA），核心包括：

- `EvoRankLoRALayer`：最大秩超空间 + 掩码激活 + 在线扩张/修剪
- `RankEvolutionController`：EMA 统计、动态阈值、计数器与冷却、Mutation 生成与提交
- `train_integration.py`：模型注入（`inject_evo_lora`）与双时间尺度训练步（`train_evo_lora_step`）
- `run_benchmark.py`：GLUE（**HuggingFace `glue` 全部 10 个配置**）/ NLG 基准脚本，支持 LoRA、AdaLoRA、EvoRank、**LoRA-GA**、**SoRA**（及 MTL-LoRA 占位说明）；支持 TensorBoard / W&B、**CSV 导出**，以及 **`--output_dir` 下按实验写入 `metrics.jsonl`、可选中间 checkpoint、训练结束 `final/`（权重 + tokenizer）与 `--verify_n_samples` 冒烟打印**

### 对比方法（`--methods`）

| 方法 | 说明 |
|------|------|
| `lora` | 标准 HuggingFace PEFT LoRA |
| `adalora` | HuggingFace PEFT AdaLoRA（含 RankAllocator 步后 `update_and_allocate`；正交正则因本脚本只取 logits 在外部 CE 上显式加入，系数 `--adalora_orth_reg_weight`，默认 0.1 与论文脚本 `reg_orth_coef` 常见设置一致；可调 `--adalora_init_r` / `--adalora_tinit` / `--adalora_tfinal` / `--adalora_delta_t`） |
| `evorank` | 本仓库 EvoRank-LoRA |
| `lora-ga` | 梯度 SVD 初始化 + 标准 PEFT LoRA；DDP 下仅 rank0 用无分布式 Sampler 的 loader 取前 `--lora_ga_batches` 个 batch，再 `broadcast_object_list` 同步 |
| `sora` | 带 `gate` 的稀疏低秩旁路；训练时需在主损失上加 L1（脚本已用 `--sora_sparse_lambda`，可选 `--sora_lambda_warmup_steps`） |
| `mtl-lora` | 未实现：需联合多任务与 task id，与当前 `--task_list` 串行协议不对齐 |

### GLUE 全任务（`--task_name`，`task_type=nlu`）

脚本通过 `load_dataset("glue", task_name)` 加载数据；**已覆盖 [HuggingFace `glue`](https://huggingface.co/datasets/glue) 的全部 10 个配置**（字段映射与验证 split 约定已内置，与官方配置名一致）：

| `task_name` | 说明 | 验证 split | 验证主指标（与 GLUE 常用约定一致，实现见 `glue_metrics.py`） |
|-------------|------|------------|----------------------------------|
| `cola` | 语言可接受性（单句） | `validation` | **Matthews 相关系数** |
| `sst2` | 情感（单句） | `validation` | **Accuracy** |
| `mrpc` | 释义等价（句对） | `validation` | **F1**（binary） |
| `qqp` | 问句等价（问句对） | `validation` | **F1**（binary） |
| `stsb` | 句对语义相似度（**回归**，标签为连续分数） | `validation` | **(Pearson + Spearman) / 2** |
| `mnli` | 自然语言推理（前提–假设） | `validation_matched` | **Accuracy** |
| `qnli` | 问答蕴含 | `validation` | **Accuracy** |
| `rte` | 文本蕴含 | `validation` | **Accuracy** |
| `wnli` | Winograd 式蕴含（极小集） | `validation` | **Accuracy** |
| `ax` | 诊断集（MNLI 风格前提–假设） | `validation` | **Accuracy** |

**注意：**

- **`stsb`**：训练为 MSE 回归；验证在**完整 dev 集**上算 Pearson 与 Spearman，主标量为二者**算术平均**（与 GLUE 总分里该任务的常见合成方式一致）。
- **DDP**：与 NLG 相同，验证指标仅在 **rank0** 上对**全量** dev 集计算，再 `broadcast` 到各卡，避免 `DistributedSampler` 子集偏差。
- CSV 列名仍为 **`best_val_accuracy`**，语义为「该任务验证主指标」（不限于 accuracy）；TensorBoard 为 **`val/<指标名>`**（如 `val/matthews_corrcoef`、`val/f1`、`val/pearson_spearman_mean`）。`metrics.jsonl` 含字段 **`glue_metric`** 标明指标键名。
- **`--task_list`** 可串行跑多个 `task_name`；一次扫**全部 10 个 GLUE 任务**（含诊断集 `ax` 与回归 `stsb`）时可写：  
  `--task_list ax cola sst2 mrpc qqp stsb mnli qnli rte wnli`（顺序可任意；耗时与磁盘占用随任务数线性增长）。

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
- GLUE：**10 个任务**验证均使用 **官方主指标**（CoLA→Matthews，MRPC/QQP→F1，STS-B→Pearson 与 Spearman 均值，其余分类任务→Accuracy），见 `glue_metrics.py`。
- 训练产物与 checkpoint（与 `--log_dir` 分离）：
  - 默认 `--output_dir artifacts`，子目录形如 `artifacts/<task>_<backbone>_<method>/`
  - `metrics.jsonl`、可选 `--save_steps` / `--save_every_epoch`、结束 `final/`（PEFT 为 `save_pretrained`；`evorank`/`sora` 为 `model_state.pt`）
  - `final/` 内同时保存 **tokenizer**，便于离线 `AutoTokenizer.from_pretrained(<final_dir>)`
  - 中间 `.pt` **仅含** unwrap 后的模型与优化器/调度器；**不含** EvoRank Controller 的 EMA 等（V1 取舍，见下文「日志与结果」）
  - `--verify_n_samples`：训练后主进程打印少量验证样本对照（默认 `2`）

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

## 安装步骤（中国网络/清华源 + 当前目录 conda 环境）

### 1) 在当前目录创建 conda 环境（不污染系统）

```bash
mkdir -p envs datasets models
conda create -y -p ./envs/evorank python=3.10
conda activate ./envs/evorank
python -m pip install --upgrade pip
```

如需 conda 包也走清华镜像，可在当前用户配置（可选）：

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

### 2) 配置 pip 清华源（当前 shell 会话）

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

### 3) 安装 PyTorch（单独安装，避免 requirements 与 CUDA 轮子冲突）

二选一（推荐先尝试 cu124）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4) 安装其余依赖

```bash
pip install -r requirements.txt
```

### 5) 缓存目录约定（已内置默认）

- 数据默认缓存到 `./datasets`
- 模型默认缓存到 `./models`

如需显式指定：

```bash
python run_benchmark.py --dataset_cache_dir datasets --model_cache_dir models
```

---

## 快速开始

**产物目录**：默认 **`--output_dir artifacts`**（与 TensorBoard 的 `--log_dir` 无关）。每个 `task×backbone×method` 会生成独立子目录，内含 `metrics.jsonl`、训练结束后的 **`final/`**（权重 + tokenizer）及控制台 **`[verify]`** 样本（默认 `--verify_n_samples 2`）。若只测速度、不写盘，可加 **`--no_output_dir`**；关闭样本打印用 **`--verify_n_samples 0`**。更多选项见「日志与结果」。

### 1) 冒烟测试（单卡，推荐先跑）

**A. 最小三法（最快）**：验证数据加载、PEFT LoRA/AdaLoRA、EvoRank 双时间尺度与 CSV。AdaLoRA 需显式正交系数（与训练循环一致）；步数很少时把 `--adalora_delta_t` 调小，避免整段训练不做预算更新。

```bash
python run_benchmark.py \
  --methods lora adalora evorank \
  --task_name sst2 \
  --model_name roberta-base \
  --max_train_steps 50 \
  --T_es 20 \
  --warmup_ratio 0.1 \
  --adalora_delta_t 50 \
  --adalora_orth_reg_weight 0.1 \
  --seed 42 \
  --log_dir runs/smoke \
  --output_dir artifacts \
  --export_csv results_smoke.csv
```

**B. 对比方法全开（较慢）**：在 A 的基础上加入 LoRA-GA（rank0 上取少量 batch 做梯度 SVD 初始化）与 SoRA（gate + L1）。

```bash
python run_benchmark.py \
  --methods lora adalora evorank lora-ga sora \
  --task_name sst2 \
  --model_name roberta-base \
  --max_train_steps 50 \
  --T_es 20 \
  --warmup_ratio 0.1 \
  --adalora_delta_t 50 \
  --adalora_orth_reg_weight 0.1 \
  --lora_ga_batches 4 \
  --sora_sparse_lambda 1e-3 \
  --sora_lambda_warmup_steps 0 \
  --seed 42 \
  --log_dir runs/smoke_full \
  --output_dir artifacts \
  --export_csv results_smoke_full.csv
```

**冒烟预期（健康检查）**：`--max_train_steps 50` 仅覆盖约一个 epoch 的一小段，**SST-2 上 `val_accuracy` 接近 0.5（随机基线）是正常现象**；多种方法数值相同或极接近也常见。应确认：无 Python 异常、各方法均打印 `val_*` 行、**`[verify]`** 有输出、`artifacts/.../final/` 与 CSV 落盘。加载 `roberta-base` 时若出现 classifier 权重「新初始化」的提示，属在下游头随机初始化上的预期行为。

若使用 W&B，在以上命令末尾追加：`--use_wandb --wandb_project <your_project>`。

冒烟结束后可检查例如 `artifacts/sst2_roberta-base_lora/final/` 与 `metrics.jsonl`，并查看终端 **`[verify]`** 行是否与标签一致。

使用 nohup / 后台跑 torchrun 前建议：

```bash
mkdir -p logs
```

### 1.1) DDP 多卡冒烟测试（torchrun）

在 Linux 多卡上先用极小步数验证：各卡 loss/同步、EvoRank trial、LoRA-GA 的 barrier/broadcast（若包含 `lora-ga`）。

**最小三法：**

```bash
torchrun --nproc_per_node=2 --master_port=29500 \
  run_benchmark.py \
  --ddp \
  --task_name sst2 \
  --model_name roberta-base \
  --methods lora adalora evorank \
  --max_train_steps 20 \
  --T_es 10 \
  --warmup_ratio 0.1 \
  --adalora_delta_t 10 \
  --adalora_orth_reg_weight 0.1 \
  --mini_val_k 4 \
  --seed 42 \
  --log_dir runs/ddp_smoke \
  --output_dir artifacts \
  --export_csv results_ddp_smoke.csv
```

若出现 `ProcessGroupNCCL::WorkNCCL::checkTimeout` / `ncclCommWatchdog`：多为某 rank **少做了一次集体通信**（历史上 EvoRank 在 `mutations` 列表长度不一致时会触发）。当前版本已在 `train_integration.py` 中对齐 trial 的 `all_reduce` 次数；若仍超时，可尝试增大 `export NCCL_TIMEOUT=1800` 或检查集群 IB/GPU 拓扑。

**对比全开（可选）**：将 `--methods` 改为 `lora adalora evorank lora-ga sora`，并追加例如 `--lora_ga_batches 2 --sora_sparse_lambda 1e-3`（DDP 下 LoRA-GA 仅在 rank0 用全数据 loader 的前若干个 batch 估计，他卡 barrier 等待）。产物与 checkpoint **仅 rank0 写盘**。

### 2) 主结果模板（多任务）

全对比时包含 LoRA-GA / SoRA；若只想跑基线，可将 `--methods` 改为 `lora adalora evorank` 并去掉 `--lora_ga_batches` / `--sora_*`。

```bash
python run_benchmark.py \
  --methods lora adalora evorank lora-ga sora \
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
  --adalora_orth_reg_weight 0.1 \
  --lora_ga_batches 8 \
  --sora_sparse_lambda 1e-3 \
  --lambda_c 0.0 \
  --complexity_mode rank_sum \
  --population_strategy all \
  --seed 42 \
  --log_dir runs/main_results \
  --output_dir artifacts \
  --export_csv results_main.csv
```

双卡（torchrun）+ nohup 后台运行：

```bash
nohup torchrun --nproc_per_node=2 --master_port=29500 \
  run_benchmark.py \
  --ddp \
  --methods lora adalora evorank lora-ga sora \
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
  --adalora_orth_reg_weight 0.1 \
  --lora_ga_batches 8 \
  --sora_sparse_lambda 1e-3 \
  --lambda_c 0.0 \
  --complexity_mode rank_sum \
  --population_strategy all \
  --seed 42 \
  --log_dir runs/main_results_ddp \
  --output_dir artifacts \
  --export_csv results_main_ddp.csv \
  > logs/main_results_ddp.out 2>&1 &
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
  --output_dir artifacts \
  --export_csv results_ablation_full.csv
```

双卡（torchrun）+ nohup 后台运行：

```bash
nohup torchrun --nproc_per_node=2 --master_port=29500 \
  run_benchmark.py \
  --ddp \
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
  --log_dir runs/ablation_full_ddp \
  --output_dir artifacts \
  --export_csv results_ablation_full_ddp.csv \
  > logs/ablation_full_ddp.out 2>&1 &
```

### 4) 效率模板

```bash
python run_benchmark.py \
  --methods lora adalora evorank lora-ga sora \
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
  --adalora_orth_reg_weight 0.1 \
  --lora_ga_batches 8 \
  --sora_sparse_lambda 1e-3 \
  --lambda_c 0.001 \
  --complexity_mode rank_sum \
  --lambda_pop 16 \
  --population_strategy all \
  --seed 42 \
  --log_dir runs/efficiency \
  --output_dir artifacts \
  --export_csv results_efficiency.csv
```

双卡（torchrun）+ nohup 后台运行：

```bash
nohup torchrun --nproc_per_node=2 --master_port=29500 \
  run_benchmark.py \
  --ddp \
  --methods lora adalora evorank lora-ga sora \
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
  --adalora_orth_reg_weight 0.1 \
  --lora_ga_batches 8 \
  --sora_sparse_lambda 1e-3 \
  --lambda_c 0.001 \
  --complexity_mode rank_sum \
  --lambda_pop 16 \
  --population_strategy all \
  --seed 42 \
  --log_dir runs/efficiency_ddp \
  --output_dir artifacts \
  --export_csv results_efficiency_ddp.csv \
  > logs/efficiency_ddp.out 2>&1 &
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
  - 默认字段：`task/backbone/method/val_metric_key/trainable_params/best_val_accuracy/peak_memory_mb/avg_active_rank/total_train_time_sec/artifact_dir/final_dir`（`val_metric_key` 标明 `best_val_accuracy` 对应何种主指标，如 `accuracy`、`f1`、`matthews_corrcoef`）
- **训练产物目录**（与 TensorBoard 的 `--log_dir` 相互独立）：
  - `--output_dir`：默认 `artifacts`；每个 `task×backbone×method` 会在其下创建子目录，例如 `artifacts/sst2_roberta-base_lora/`。
  - `--no_output_dir`：关闭该目录下所有落盘（不写 `metrics.jsonl`、checkpoint、`final/`）。
  - `metrics.jsonl`：每 epoch 一行 JSON（`epoch`、`global_step`、`val_metric`、`best_val`、`train_loss_ema`；NLU 另有 **`glue_metric`** 标明主指标键名）。
  - `--save_steps N`：`N>0` 时每 N 个训练 step 保存 `checkpoint_step_*.pt`（仅 rank0；内含 unwrap 后的 `inner` 权重与优化器/调度器，**不含** EvoRank Controller 的 EMA 等内部状态）。
  - `--save_every_epoch`：每个 epoch 末保存 `checkpoint_epoch_*.pt`。
  - `--no_save_final_model`：训练结束不写 `final/`（默认会写）。
  - `final/`：**自包含推理资产** — PEFT 方法调用 `save_pretrained`（含 `adapter_config.json`）；`evorank`/`sora` 等手写注入保存 `model_state.pt`；并写入 `training_meta.json` 与 **`tokenizer` 文件**（便于 `AutoTokenizer.from_pretrained(<final_dir>)` 离线加载）。
  - `--verify_n_samples K`：训练结束后主进程打印验证集前 K 条分类 `[Gold]` vs `[Pred]`，或 NLG 一条摘要片段；`K=0` 关闭（默认 `2`）。

---

## NLG（生成任务）示例：CNN/DailyMail + ROUGE-L
脚本支持 `--task_type nlg`。当前默认使用 CNN/DailyMail；验证阶段在单卡或 DDP 下 rank0 上对**完整验证集**（或 `--nlg_eval_max_samples` 截断）算 ROUGE-L，他卡 `barrier` 同步。

```bash
python run_benchmark.py \
  --task_type nlg \
  --nlg_dataset_name cnn_dailymail \
  --task_name cnn_dailymail \
  --model_name t5-small \
  --methods lora adalora evorank lora-ga sora \
  --target_rank 8 \
  --epochs 1 \
  --batch_size 4 \
  --max_length 256 \
  --max_target_length 64 \
  --generation_max_new_tokens 64 \
  --nlg_eval_max_samples 50 \
  --max_train_steps 50 \
  --T_es 20 \
  --warmup_ratio 0.1 \
  --adalora_delta_t 50 \
  --adalora_orth_reg_weight 0.1 \
  --lora_ga_batches 4 \
  --sora_sparse_lambda 1e-3 \
  --lambda_c 0.0 \
  --complexity_mode rank_sum \
  --lambda_pop 16 \
  --population_strategy all \
  --seed 42 \
  --log_dir runs/nlg_smoke \
  --output_dir artifacts \
  --export_csv results_nlg_smoke.csv
```

仅跑轻量基线时可改为 `--methods lora adalora evorank` 并去掉 `--lora_ga_batches`、`--sora_sparse_lambda`。

---

## 常见问题

### 1) AdaLoRA 报错：`total_step is None`

确保 `peft_factory` 初始化 AdaLoRA 时传入了 `total_step`。本仓库已做兼容处理（自动按计划步数注入）。

### 2) EvoRank 精度接近随机

检查分类头是否解冻。HF 的 LoRA/AdaLoRA 在 `SEQ_CLS` 任务通常会处理任务头；EvoRank 手动注入需要显式解冻 `classifier`/`score` 参数。本仓库已处理。

### 3) 报设备不一致（CPU/CUDA）

典型报错：`Expected all tensors to be on the same device`。本仓库在 `train_evo_lora_step` 中已加入 controller 状态张量对齐。

### 4) 如何从 `final/` 加载 PEFT 适配器

`final/` 里是 **adapter**（及 tokenizer），推理时仍需原始 **`AutoModel*.from_pretrained(<base_model>)`**，再 **`PeftModel.from_pretrained(base_model, <final_dir>)`**（或与 `AutoModelForSequenceClassification` 等组合，按 HF PEFT 文档）。`evorank`/`sora` 的 `model_state.pt` 对应手写注入后的 `inner` 权重，加载方式与训练时构图一致，需自行 `load_state_dict`。

### 5) 中间 checkpoint 能否完美续训 EvoRank

当前 `.pt` **不保存** `RankEvolutionController` 的 EMA/计数器等；续训仅保证优化器与 `inner` 权重一致，ES 演化轨迹可能与「从头连续跑」不同。以推理与打表为主时通常足够；若需 bit 级续训一致，需在后续版本为 Controller 增加 `state_dict` 并写入同一 checkpoint。

---

## 主要脚本

- `evo_rank_lora.py`：可演化 LoRA 层
- `rank_evolution_controller.py`：演化控制器与 Mutation 体系
- `train_integration.py`：注入与双时间尺度训练
- `glue_metrics.py`：GLUE 全 10 任务验证主指标（Matthews / Acc / F1 / Pearson–Spearman 均值等）
- `run_benchmark.py`：主实验入口
