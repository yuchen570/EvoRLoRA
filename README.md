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
| `lora-ga` | 梯度 SVD 初始化 + 标准 PEFT LoRA；DDP 下仅 rank0 用无分布式 Sampler 的 loader 取前 `--lora_ga_batches` 个 batch，再 `broadcast_object_list` 同步；支持 `--lora_ga_use_rslora`（rsLoRA 缩放）与 `--lora_ga_stable_gamma`（stable init） |
| `sora` | 带 `gate` 的稀疏低秩旁路；训练时需在主损失上加 L1（脚本已用 `--sora_sparse_lambda`，可选 `--sora_lambda_warmup_steps` / `--sora_lambda_schedule`） |
| `mtl-lora` | 未实现：需联合多任务与 task id，与当前 `--task_list` 串行协议不对齐 |

### 原始文献实验与参数考量

在我们对比的主流低秩演化、梯度预估等方法时，调研了各对比算法在其原始代码仓库/论文中进行的实验范围与核心配置参数：
- **AdaLoRA**：主要在 GLUE（如选用 DeBERTaV3-base）与一些特定的生成式任务（如 XSum、SQuADv2，选用 BART-large 或 DeBERTaV3）上进行了验证。**参数特点**：初始探测秩 `lora_r` 往往设为目标保留秩 `target_rank` 的 1.5 倍或数倍，正交正则化系数 `reg_orth_coef=0.1`，并采用基于指数移动平均(EMA)的预热调度（`init_warmup`, `final_warmup`, `mask_interval`）。
- **LoRA-GA**：重点围绕大型语言模型（如 Llama 2-7B on MetaMath QA / Wizard-LM）的浮点与 4-bit/8-bit 量化训练开展，同时也涉及了部分 GLUE（如 T5-base on SST-2）。**参数特点**：利用完整前向数据包通过 `estimate_gradient` 函数预估梯度，并直接将特征向量作为参数 SVD 初始化使用，一般使用 `lora_alpha` 缩放，免除过多的运行时复杂调参。
- **SoRA**：在 GLUE 上将自身与基础模型的全参数微调 (Full Fine-Tune)、Adapter 以及 BitFit 进行了详尽的基准对比。**参数特点**：在 Loss 上增加了带阈值的稀疏正则化惩罚，核心参数为 `sparse_lambda`（对应论文中的 $\eta_t$）、`sparse_lambda_2`（对应 $\xi$），初始最大探测秩 `lora_r`（$r_{max}$），以及动态稀疏退火策略如 `linear` 并附带相应的更新步数设定（`lambda_num`）。

> [!NOTE]
> **验证策略说明**：尽管上述对比算法在论文中还执行了预训练大语言模型（LLM）指令微调、量化训练、或者与全参数微调、BitFit / Adapter 比较，**但在我们的 EVO 框架（EvoRLoRA）评估体系下，只需要跑其余的这些 PEFT 同类对比算法（`lora`, `adalora`, `lora-ga`, `sora`）并与 EvoRank 进行同台对比即可充分验证 EvoRank 的效果**。也就是直接使用我们在各个脚本中统一的验证协议进行评测即可，无需去复刻它们全参或量化的环境。

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
| `ax` | 诊断集（MNLI 风格；见下 **HF 无金标**） | — | **Accuracy**（需自带金标数据源） |

**注意：**

- **`ax`（HuggingFace `datasets` 的 `glue`/`ax`）**：`test` **标签全部为 `-1`（不公开金标）**，无法用本脚本的交叉熵训练或计算验证主指标；若指定 `--task_name ax` 或把 `ax` 写入 `--task_list`，`setup_data_and_model` 会 **显式报错**。GLUE「10 个配置」在 Hub 上均可 `load_dataset`，但 **有监督跑法仅适用于其余 9 项**；一键多任务命令里请使用 **`cola sst2 mrpc qqp stsb mnli qnli rte wnli`**（不要含 `ax`），除非自行提供带金标的 AX 数据管线。
- **`stsb`**：训练为 MSE 回归；验证在**完整 dev 集**上算 Pearson 与 Spearman，主标量为二者**算术平均**（与 GLUE 总分里该任务的常见合成方式一致）。
- **DDP**：与 NLG 相同，验证指标仅在 **rank0** 上对**全量** dev 集计算，再 `broadcast` 到各卡，避免 `DistributedSampler` 子集偏差。
- CSV 列名仍为 **`best_val_accuracy`**，语义为「该任务验证主指标」（不限于 accuracy）；TensorBoard 为 **`val/<指标名>`**（如 `val/matthews_corrcoef`、`val/f1`、`val/pearson_spearman_mean`）。`metrics.jsonl` 含字段 **`glue_metric`** 标明指标键名。
- **`--task_list`** 可串行跑多个 `task_name`。**有监督训练/验证**请使用 **9 项**（勿含 `ax`，原因见上表 `ax` 说明）：  
  `--task_list cola sst2 mrpc qqp stsb mnli qnli rte wnli`（顺序可任意）。`ax` 仍列在表中仅表示 Hub 配置存在，与本脚本可跑任务集合不同。

---

## 近期更新（与论文实现一致性相关）

当前代码已支持以下关键机制：

- 动态阈值（分位数）+ EMA 平滑
- 持久计数器（`H_g` / `H_p`）+ cooldown
- Rolling Candidacy（落选候选计数保留）
- DDP Trial 奖励同步（`all_reduce`）
- no-op 候选参与 elitist 选择
- Reward 复杂度正则：`-lambda_c * C(z)`，支持 `rank_sum` / `size_aware`
- ES population 子采样：`--lambda_pop` / `--population_strategy all|random`
- 学术协议批量运行：`--task_list` / `--model_list` / `--export_csv`
- GLUE：Hub 上 **10 个配置**均可加载；**有监督验证**为 **9 项**（不含 `ax`），均使用 **官方主指标**（CoLA→Matthews，MRPC/QQP→F1，STS-B→Pearson 与 Spearman 均值，其余分类→Accuracy），见 `glue_metrics.py`
- 训练产物与 checkpoint（与 `--log_dir` 分离）：`--output_dir artifacts`，子目录 `artifacts/<task>_<backbone>_<method>/`，含 `metrics.jsonl`、可选 checkpoint、`final/`（PEFT `save_pretrained` 或 `model_state.pt` + tokenizer）

**最新基准测试修复与优化 (2025/3)：**
- **分类头特征隔离学习率 (`--head_lr`)**：彻底修复由于 `lr=2e-5` 在 CoLA / MRPC 上导致随机初始化的分类头无法收敛的问题。默认强制将 classifier/score 头推升至 `max(lr, 5e-4)` 从而正常获取非随机准确率，不再需要为头专门改总 `lr` 或者改 loss_func。
- **AdaLoRA 预算调度充分化 (`tfinal`)**：由于 PEFT 参数设计的含糊性，先前代码将 `0.8*total_steps` 用于锁死预算期（导致前 20% 只草草分配）。已修改默认值为 `0.1*total_steps`，让 AdaLoRA 有前 90% 的生命周期进行动态调整。
- **SoRA 近端梯度平滑 (`sora_sparse_lambda_2`)**：默认参数下调至对齐原论文主线稳定性的 `3e-4` 级别（原 `1e-3`，经常在训练中后期激进裁剪导致表达能力崩溃）。

**论文对齐新增参数（实验对齐 spec 已全部落地）：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora_alpha` | `None`（→ `2*r`） | 独立设置 LoRA alpha；NLU 论文对齐设 `16`，NLG/BART 设 `32` |
| `--target_modules` | `None`（自动推断） | 逗号分隔的注入模块后缀，如 `query_proj,key_proj,value_proj,intermediate.dense,output.dense` |
| `--max_grad_norm` | `None`（不裁剪） | 梯度裁剪阈值；SoRA 论文使用 `0.1` |
| `--seed_list` | `None` | 多种子串行运行，如 `--seed_list 0 21 42 81 100`；CSV 自动追加均值/标准差汇总行 |
| `--lora_ga_use_rslora` | `False` | LoRA-GA 启用 rsLoRA 缩放（`lora_alpha/sqrt(r)`），对标官方 reproduce 配置 |
| `--lora_ga_stable_gamma` | `None` | LoRA-GA stable init gamma（官方默认 64）；指定后 SVD 初始化按 `gamma * ‖W‖_F / (sqrt(r) * S[0])` 缩放 |
| `--sora_lambda_schedule` | `None` | SoRA lambda 调度策略（如 `linear`）；`None` 为 no-schedule 主线 |
| `--sora_max_lambda` | `10.0` | SoRA schedule 最大 lambda |
| `--sora_lambda_num` | `5` | SoRA schedule 步数 |

**各 backbone 自动推断 target_modules：**

| backbone | 自动推断默认值 | 论文 6 类模块（`--target_modules` 显式传入） |
|---|---|---|
| DeBERTa-v3-base | `query_proj,key_proj,value_proj` | `query_proj,key_proj,value_proj,intermediate.dense,output.dense` |
| RoBERTa/BERT | `query,value` | `query,key,value,intermediate,output` |
| BART (NLG) | `q_proj,v_proj` | `q_proj,k_proj,v_proj,out_proj,fc1,fc2` |
| T5 | `q,v` | — |

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

**对比全开（DDP 冒烟）**：在「最小三法」相同步数与 AdaLoRA 参数下，增加 `lora-ga`、`sora` 及对应 CLI。DDP 下 LoRA-GA 仅在 rank0 用全数据 loader 的前若干个 batch 做梯度 SVD 初始化，他卡 barrier 等待；产物与 checkpoint **仅 rank0 写盘**。

```bash
torchrun --nproc_per_node=2 --master_port=29500 \
  run_benchmark.py \
  --ddp \
  --task_name sst2 \
  --model_name roberta-base \
  --methods lora adalora evorank lora-ga sora \
  --max_train_steps 20 \
  --T_es 10 \
  --warmup_ratio 0.1 \
  --adalora_delta_t 10 \
  --adalora_orth_reg_weight 0.1 \
  --lora_ga_batches 2 \
  --sora_sparse_lambda 1e-3 \
  --mini_val_k 4 \
  --seed 42 \
  --log_dir runs/ddp_smoke_full \
  --output_dir artifacts \
  --export_csv results_ddp_smoke_full.csv
```

### 2) 主结果模板（多任务）

**说明：** 下列命令用于 **RoBERTa-base + 常见微调超参**（`lr=2e-5`、`epochs=3`、`weight_decay=0.01`）下的 **方法同台对比**，**不是** AdaLoRA / SoRA 等论文原文的表格数字复现。论文级协议见下文 **「AdaLoRA / SoRA 论文复现（DeBERTa-v3-base，GLUE NLU）」** 等专节。

> [!TIP]
> **关于不同 GLUE 任务的 Epoch 设置建议**
> 根据评测日志分析，统一设置 3 个或 5 个 Epoch 对于所有任务**是不合理**的。不同任务应设置不同的 Epoch 才能确保充分训练以及稳定的数值表现：
> - **严重欠拟合的任务（需 15 ~ 20 Epochs）**：`stsb`, `cola`, `rte`。这些子集数据量小且所需更新步数少，如果在 5 个 Epoch 内停止会导致明显欠拟合或严重震荡。
> - **极小数据集易波动任务（需 10 ~ 20 Epochs）**：`mrpc`, `wnli`。非常容易出现大幅震荡或直接崩溃（如 WNLI 预测多数类别）。建议充分 Warmup 并放长训练轮数，最好配合 Early Stopping。
> - **大数据集易收敛的任务（3 ~ 5 Epochs 即可）**：`sst2`, `mnli`, `qnli`, `qqp`。这些数据量大的任务在 3~5 轮时通常已展现完美的收敛平台，为了最高精度最多尝试 5~10 轮。
> 如果你需要同时对所有子集跑 `--task_list ...`，建议为大规模和小规模数据集分别设置不同的训练脚本。

全对比时包含 LoRA-GA / SoRA；若只想专注跑基线，可将 `--methods` 改为 `lora adalora evorank` 并去掉 `--lora_ga_batches / --sora_sparse_lambda*` 等分支参数。框架内部已经为 `sora` 和 `lora-ga` 等强约束算法做好了动态超参兼容（如自动切换 `max_grad_norm=0.1` 和免衰减修正），因此**您可以在同一个命令中横向并跑这些算法**，而完全不需要手动切碎脚本调整那些苛刻的论文约束。
# === 脚本 A: 大数据集 (3~5 Epochs 即可收敛) ===
双卡（torchrun）+ nohup 后台运行：

```bash
nohup torchrun --nproc_per_node=2 --master_port=29500 \
  run_benchmark.py \
  --ddp \
  --methods lora adalora evorank lora-ga sora \
  --task_list sst2 mnli qnli qqp \
  --model_list roberta-base \
  --target_rank 8 \
  --epochs 5 \
  --batch_size 16 \
  --lr 5e-4 \
  --weight_decay 0.1 \
  --warmup_ratio 0.06 \
  --max_grad_norm 0.1 \
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
  --log_dir runs/main_results_large_ddp \
  --output_dir artifacts \
  --export_csv results_main_large_ddp.csv \
  > logs/main_results_large_ddp.out 2>&1 &
```

# === 脚本 B: 中小数据集 (需 15~20 Epochs 充分训练) ===
双卡（torchrun）+ nohup 后台运行：

```bash
nohup torchrun --nproc_per_node=2 --master_port=29500 \
  run_benchmark.py \
  --ddp \
  --methods lora adalora evorank lora-ga sora \
  --task_list cola mrpc stsb rte wnli \
  --model_list roberta-base \
  --target_rank 8 \
  --epochs 20 \
  --batch_size 32 \
  --lr 5e-4 \
  --weight_decay 0.1 \
  --warmup_ratio 0.06 \
  --max_grad_norm 0.1 \
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
  --log_dir runs/main_results_small_ddp \
  --output_dir artifacts \
  --export_csv results_main_small_ddp.csv \
  > logs/main_results_small_ddp.out 2>&1 &
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

# === 脚本 A: 大数据集 ===
双卡（torchrun）+ nohup 后台运行：

```bash
nohup torchrun --nproc_per_node=2 --master_port=29500 \
  run_benchmark.py \
  --ddp \
  --methods lora adalora evorank lora-ga sora \
  --task_list sst2 mnli qnli qqp \
  --model_list roberta-base \
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
  --log_dir runs/efficiency_large_ddp \
  --output_dir artifacts \
  --export_csv results_efficiency_large_ddp.csv \
  > logs/efficiency_large_ddp.out 2>&1 &
```

  - `--save_every_epoch`：每 epoch 结束保存 `checkpoint_epoch_*.pt`。
  - `--no_save_final_model`：训练结束不写 `final/`（默认会写）。
  - `final/`：**自包含推理资产** — PEFT 方法调用 `save_pretrained`（含 `adapter_config.json`）；`evorank`/`sora` 等手写注入保存 `model_state.pt`；并写入 `training_meta.json` 与 **`tokenizer` 文件**（便于 `AutoTokenizer.from_pretrained(<final_dir>)` 离线加载）。
  - `--verify_n_samples K`：训练结束后主进程打印验证集前 K 条分类 `[Gold]` vs `[Pred]`，或 NLG 一条摘要片段；`K=0` 关闭（默认 `2`）。

---



## NLG（生成任务）示例：CNN/DailyMail / XSum + ROUGE-1/2/L

脚本支持 `--task_type nlg`，验证阶段同时计算并记录 **ROUGE-1、ROUGE-2、ROUGE-L** 三项指标（TensorBoard `val/rouge1`、`val/rouge2`、`val/rougeL`；CSV 中的专属列）。

支持数据集：

| `--nlg_dataset_name` | 文本字段 | 目标字段 | 对应论文 |
|---|---|---|---|
| `cnn_dailymail` | `article` | `highlights` | AdaLoRA Table 2 |
| `xsum` | `document` | `summary` | AdaLoRA Table 3 |

**CNN/DailyMail 冒烟：**

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
  --seed 42 \
  --log_dir runs/nlg_smoke \
  --output_dir artifacts \
  --export_csv results_nlg_smoke.csv
```

双卡（torchrun）+ nohup 后台运行：

```bash
nohup torchrun --nproc_per_node=2 --master_port=29500 \
  run_benchmark.py \
  --ddp \
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
  --seed 42 \
  --log_dir runs/nlg_smoke_ddp \
  --output_dir artifacts \
  --export_csv results_nlg_smoke_ddp.csv \
  > logs/nlg_smoke_ddp.out 2>&1 &
```

**以 AdaLoRA 参数配置严格对齐验证 EvoRank（XSum，BART-large）：**

依据 AdaLoRA 开源库配置，XSum 上总批次大小通常为 64，学习率 5e-4，训练 25 个 epoch，首 3000 步 warmup。此处**我们仅运行 EvoRank (`--methods evorank`)** 来直接展现实效：

```bash
nohup torchrun --nproc_per_node=2 --master_port=29500 \
  run_benchmark.py \
  --ddp \
  --task_type nlg \
  --nlg_dataset_name xsum \
  --task_name xsum \
  --model_name facebook/bart-large \
  --methods evorank \
  --target_rank 8 \
  --lora_alpha 32 \
  --target_modules q_proj,k_proj,v_proj,out_proj,fc1,fc2 \
  --epochs 25 \
  --batch_size 64 \
  --max_length 768 \
  --max_target_length 64 \
  --generation_max_new_tokens 64 \
  --lr 5e-4 \
  --weight_decay 0.01 \
  --warmup_steps 3000 \
  --seed 42 \
  --log_dir runs/evorank_align_xsum_ddp \
  --output_dir artifacts \
  --export_csv results_evorank_align_xsum_ddp.csv \
  > logs/evorank_align_xsum_ddp.out 2>&1 &
```

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

## 公平对比规范（AdaLoRA / LoRA-GA / SoRA / EvoRank）

### GLUE 任务与评估指标一览

| 任务 | 全称 | 类型 | 核心目标 | 评估指标 |
|------|------|------|----------|----------|
| CoLA | Corpus of Linguistic Acceptability | 单句二分类 | 判断句子是否符合英语语法规范 | Matthews Correlation (MCC) |
| SST-2 | Stanford Sentiment Treebank | 单句二分类 | 电影评论情感二分类 | Accuracy |
| MRPC | Microsoft Research Paraphrase Corpus | 句对二分类 | 判断两句是否语义等价 | Accuracy + F1 |
| QQP | Quora Question Pairs | 句对二分类 | 判断问题对是否语义重复 | Accuracy + F1 |
| STS-B | Semantic Textual Similarity Benchmark | 句对回归 | 句对语义相似度打分 (0-5) | Pearson + Spearman |
| MNLI | Multi-Genre Natural Language Inference | 句对三分类 | 蕴含 / 中立 / 矛盾 | Accuracy |
| QNLI | Question Natural Language Inference | 句对二分类 | 判断句子是否包含问题答案 | Accuracy |
| RTE | Recognizing Textual Entailment | 句对二分类 | 判断前提是否蕴含假设 | Accuracy |
| WNLI | Winograd Natural Language Inference | 句对二分类 | 代词指代消歧 + 蕴含判断 | Accuracy |

> `run_benchmark.py` 通过 `glue_metrics.py` 自动为每个任务选择对应的官方主指标（CSV 列 `val_metric_key`），无需手动指定。

### 统一参数来源

公平对比的统一参数从三个对比算法开源库中取**交集 / 最大公约数**：

| 参数 | 取值 | 来源说明 |
|------|------|----------|
| `model` | `microsoft/deberta-v3-base` | AdaLoRA / SoRA 均以此为主实验 backbone |
| `target_rank` | 8 | SoRA `lora_r=8` |
| `lora_alpha` | 16 | SoRA / LoRA-GA reproduce 均用 16 |
| `target_modules` | 5 类 | AdaLoRA 6 类模块 PEFT 映射为 5 个 |
| `lr` | 8e-4 | SoRA 默认 |
| `batch_size` | 8 | SoRA 默认 (RTE 除外用 32) |
| `epochs` | 20 | SoRA 默认 |
| `warmup_ratio` | 0.06 | SoRA 默认 |
| `weight_decay` | 0.1 | SoRA 默认 |
| `max_grad_norm` | 0.1 | SoRA 默认 |
| `seed_list` | 0 21 42 81 100 | SoRA 官方 5 种子 |
| `sora_sparse_lambda` | 10 | SoRA `sparse_lambda=10` |
| `sora_sparse_lambda_2` | 3e-4 | SoRA `sparse_lambda_2=3e-4` |
| `lora_ga_stable_gamma` | 16 | LoRA-GA reproduce `stable_gamma=16` |

> [!IMPORTANT]
> **RTE 任务例外**：数据集仅 ~2.5k 样本，AdaLoRA 和 SoRA 均使用 `lr=1.2e-3, epochs=50, bsz=32`。单独提供 `fair_glue_deberta_rte.sh`。

### `scripts/` 公平对比脚本

| 脚本 | 场景 | 说明 |
|------|------|------|
| `fair_glue_deberta_smoke.sh` | SST-2 冒烟 | 50 步快速验证全方法正常运行 |
| `fair_glue_deberta.sh` | GLUE 8 项主实验 | 5 种子 x 5 方法 x 8 任务(不含RTE), 20 epochs |
| `fair_glue_deberta_rte.sh` | RTE 单项 | 50 epochs / lr=1.2e-3 / bsz=32 |
| `fair_nlg_xsum.sh` | XSum NLG | BART-large, 25 epochs, lora_alpha=32 |
| `fair_nlg_cnndailymail.sh` | CNN/DM NLG | BART-large, 15 epochs, lora_alpha=32 |

所有脚本均使用 `--methods lora adalora evorank lora-ga sora` 一次性横跑全部方法，保证控制变量公平。

### 公平对比最小原则

1. 同一数据与切分：同一个 `task_name` / `task_list`、同一个 `task_type`。
2. 同一训练预算：`epochs`、`max_train_steps`、`batch_size`、`seed`/`seed_list` 一致。
3. 同一优化框架：`lr`、`warmup_ratio`、`weight_decay`、`max_length` 一致；方法特有参数仅作为该方法内部自由度。
4. 同一评估口径：读取 CSV 的任务主指标列（由 `val_metric_key` 指示）对比，不混用不同指标。

CSV 会自动追加每个 `task×backbone×method` 的 `mean/std` 行，可直接用于横向比较与填表。

---

## 主要脚本

- `evo_rank_lora.py`：可演化 LoRA 层
- `rank_evolution_controller.py`：演化控制器与 Mutation 体系
- `train_integration.py`：注入与双时间尺度训练
- `lora_ga_init.py`：LoRA-GA 梯度 SVD 初始化（含 `stable_gamma` 支持）
- `glue_metrics.py`：GLUE 各子集验证主指标（Matthews / Acc / F1 / Pearson-Spearman 均值等）
- `run_benchmark.py`：主实验入口
- `scripts/fair_glue_deberta.sh`：GLUE 8 项全方法横向公平对比（DDP + 5 种子）
- `scripts/fair_glue_deberta_rte.sh`：RTE 单项全方法横向公平对比（特殊超参）
- `scripts/fair_nlg_xsum.sh`：XSum 全方法横向公平对比
- `scripts/fair_nlg_cnndailymail.sh`：CNN/DailyMail 全方法横向公平对比
- `scripts/fair_glue_deberta_smoke.sh`：冒烟测试（SST-2, 50 步）
