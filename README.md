# EvoRank-LoRA

本仓库实现了一个可演化的 LoRA 框架（EvoRank-LoRA），核心包括：

- `EvoRankLoRALayer`：最大秩超空间 + 掩码激活 + 在线扩张/修剪
- `RankEvolutionController`：EMA 统计、动态阈值、计数器与冷却、Mutation 生成与提交
- `train_integration.py`：模型注入（`inject_evo_lora`）与双时间尺度训练步（`train_evo_lora_step`）
- `run_benchmark.py`：GLUE（**HuggingFace `glue` 全部 10 个配置**）/ NLG 基准脚本，支持 LoRA、AdaLoRA、PiSSA、EvoRank、**SoRA**、Flat-LoRA、TopLoRA（及 MTL-LoRA 占位说明）；支持 TensorBoard / W&B、**CSV 导出**，以及 **`--output_dir` 下按实验写入 `metrics.jsonl`、可选中间 checkpoint、训练结束 `final/`（权重 + tokenizer）与 `--verify_n_samples` 冒烟打印**
- `run_qa_benchmark.py`：SQuAD v1.1 / v2.0 抽取式 QA（EM/F1），与 `scripts/fair_qa_squadv*.sh`、`scripts/generate_qa_table.py` 配套
- `run_nlg_benchmark.py`：Causal LM SFT（PiSSA 风格数据），与 `scripts/nlg_pissa_benchmark.sh`、`scripts/eval_nlg_pissa.py` 配套
- **`scripts/` 下各公平实验与制表**：见下文 **「`scripts/` 实验脚本与运行方式」**

### 对比方法（`--methods`）

| 方法 | 说明 |
|------|------|
| `lora` | 标准 HuggingFace PEFT LoRA |
| `adalora` | HuggingFace PEFT AdaLoRA（含 RankAllocator 步后 `update_and_allocate`；正交正则因本脚本只取 logits 在外部 CE 上显式加入，系数 `--adalora_orth_reg_weight`，默认 0.1 与论文脚本 `reg_orth_coef` 常见设置一致；可调 `--adalora_init_r` / `--adalora_tinit` / `--adalora_tfinal` / `--adalora_delta_t`） |
| `pissa` | HuggingFace PEFT PiSSA：`LoraConfig(init_lora_weights="pissa")`，在注入阶段执行主成分初始化（SVD）并保持与 LoRA 相同训练循环 |
| `evorank` | 本仓库 EvoRank-LoRA；扩张初始化见 `--expand_init_mode`（`zero` / `gradient`，后者对应论文 Proposition 3.2） |
| `sora` | 带 `gate` 的稀疏低秩旁路；训练时需在主损失上加 L1（脚本已用 `--sora_sparse_lambda`，可选 `--sora_lambda_warmup_steps` / `--sora_lambda_schedule`） |
| `flatlora` | Flat-LoRA (ICML 2025)：基于过滤方向二范数的贝叶斯期望高斯噪声注入；通过 Hook 实现在微批次内实时扰动，支持 DDP 同步；默认扰动强度 `--flatlora_rho 0.05` |
| `toplora` | TopLoRA (NeurIPS 2025)：引入 token-wise 奇异值缩放 (TopSingularValue)；秩固定；不支持 merge；默认使用 `--toplora_dropout 0.05` |
| `mtl-lora` | 未实现：需联合多任务与 task id，与当前 `--task_list` 串行协议不对齐 |

### 原始文献实验与参数考量

在我们对比的主流低秩演化、梯度预估等方法时，调研了各对比算法在其原始代码仓库/论文中进行的实验范围与核心配置参数：
- **AdaLoRA**：主要在 GLUE（如选用 DeBERTaV3-base）与一些特定的生成式任务（如 XSum、SQuADv2，选用 BART-large 或 DeBERTaV3）上进行了验证。**参数特点**：初始探测秩 `lora_r` 往往设为目标保留秩 `target_rank` 的 1.5 倍或数倍，正交正则化系数 `reg_orth_coef=0.1`，并采用基于指数移动平均(EMA)的预热调度（`init_warmup`, `final_warmup`, `mask_interval`）。
- **SoRA**：在 GLUE 上将自身与基础模型的全参数微调 (Full Fine-Tune)、Adapter 以及 BitFit 进行了详尽的基准对比。**参数特点**：在 Loss 上增加了带阈值的稀疏正则化惩罚，核心参数为 `sparse_lambda`（对应论文中的 $\eta_t$）、`sparse_lambda_2`（对应 $\xi$），初始最大探测秩 `lora_r`（$r_{max}$），以及动态稀疏退火策略如 `linear` 并附带相应的更新步数设定（`lambda_num`）。
- **TopLoRA (NeurIPS 2025)**：提出 token-wise 的输入输出投影，通过 λ(x) = exp(RMSNorm(x @ W_λ)) 实现逐 token 的奇异值缩放。**参数特点**：秩固定，额外引入 `d_in * r` 的参数量。默认 dropout 0.05。
- **Flat-LoRA (ICML 2025)**：针对 LoRA 容易陷入尖锐极小值导致泛化能力下降的问题，提出基于参数矩阵行方向范数缩放的贝叶斯正则化。**参数特点**：主要引入扰动强度参数 `rho` (对应脚本 `--flatlora_rho`，原始文献建议默认 0.05)。本仓库在实现上进行了大幅加固，采用 Hook + CPU 缓存实时重演的方式注入高斯噪声，从根本上免疫了因大微批（梯度累加）以及多卡同步（DDP 并行）可能带来的数学性质崩坏与梯度发散问题。

> [!NOTE]
> **验证策略说明**：尽管上述对比算法在论文中还执行了预训练大语言模型（LLM）指令微调、量化训练、或者与全参数微调、BitFit / Adapter 比较，**但在我们的 EVO 框架（EvoRLoRA）评估体系下，只需要跑这些 PEFT 同类对比算法（`lora`, `adalora`, `pissa`, `sora`）并与 EvoRank 进行同台对比即可充分验证 EvoRank 的效果**。也就是直接使用我们在各个脚本中统一的验证协议进行评测即可，无需去复刻它们全参或量化的环境。

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
| `wnli` | Winograd 式蕴含（极小集，不建议） | `validation` | **Accuracy** |
| `ax` | 诊断集（MNLI 风格；见下 **HF 无金标**） | — | **Accuracy**（需自带金标数据源） |
**注意：**

- **`ax`（HuggingFace `datasets` 的 `glue`/`ax`）**：`test` **标签全部为 `-1`（不公开金标）**，无法用本脚本的交叉熵训练或计算验证主指标；若指定 `--task_name ax` 或把 `ax` 写入 `--task_list`，`setup_data_and_model` 会 **显式报错**。GLUE「10 个配置」在 Hub 上均可 `load_dataset`，但 **有监督跑法仅适用于其余 8 项**；一键多任务命令里请使用 **`cola sst2 mrpc qqp stsb mnli qnli rte`**（不要含 `ax`）。
- **`stsb`**：训练为 MSE 回归；验证在**完整 dev 集**上算 Pearson 与 Spearman，主标量为二者**算术平均**（与 GLUE 总分里该任务的常见合成方式一致）。
- **DDP**：与 NLG 相同，验证指标仅在 **rank0** 上对**全量** dev 集计算，再 `broadcast` 到各卡，避免 `DistributedSampler` 子集偏差。
- CSV 列名仍为 **`best_val_accuracy`**，语义为「该任务验证主指标」（不限于 accuracy）；TensorBoard 为 **`val/<指标名>`**（如 `val/matthews_corrcoef`、`val/f1`、`val/pearson_spearman_mean`）。`metrics.jsonl` 含字段 **`glue_metric`** 标明指标键名。
- **`--task_list`** 可串行跑多个 `task_name`。**有监督训练/验证**请使用 **8 项**（勿含 `ax`；也不建议含 `wnli`，数据集过小）：
  `--task_list cola sst2 mrpc qqp stsb mnli qnli rte`（顺序可任意）。`ax`,`wnli` 仍列在表中仅表示 Hub 配置存在，与本脚本可跑任务集合不同。

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
- GLUE：Hub 上 **10 个配置**均可加载；**有监督验证**默认使用 **8 项**（不含 `ax`；也不建议含 `wnli`，数据集过小），均使用 **官方主指标**（CoLA→Matthews，MRPC/QQP→F1，STS-B→Pearson 与 Spearman 均值，其余分类→Accuracy），见 `glue_metrics.py`
- EvoRank：`run_benchmark.py` 暴露 `--evorank_r_max`（每层秩超空间上限，默认 16）、`--evo_alpha_u` / `--evo_beta_u`（容量统计组合系数，默认均为 1.0）、`--expand_init_mode zero|gradient`（扩张初始化：`zero` 为 B 列 cold start；`gradient` 为论文 Proposition 3.2，在 backward 后缓存的投影梯度上做 power iteration 得主奇异方向）；`--target_rank` 对应注入时的初始活跃秩 `r_init`。每层演化上下界还与控制器内 `r_min=2` 及 `r_max=evorank_r_max` 一致
- 双时间尺度步内：`train_evo_lora_step` 在 **`clip_grad_norm_` 之前** 写入层统计/梯度缓存，再执行裁剪与 `optimizer.step()`，与当前实现一致
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
| `--flatlora_rho` | `0.05` | Flat-LoRA 独占参数：基于高斯噪声对基础权重的动态扰动强度系数 |
| `--sora_lambda_schedule` | `None` | SoRA lambda 调度策略（如 `linear`）；`None` 为 no-schedule 主线 |
| `--sora_max_lambda` | `10.0` | SoRA schedule 最大 lambda |
| `--sora_lambda_num` | `5` | SoRA schedule 步数 |
| `--evorank_r_max` | `16` | EvoRank 每层最大秩上限 `R_max`（超空间宽度）；须 ≥ `--target_rank` |
| `--evo_max_reallocate_candidates` | `8` | 仅 `evorank`：控制 Reallocation 候选限流与 top-k cross 的强约束配置，用于防御跨层重分配组合爆炸 |
| `--evo_alpha_u` | `1.0` | 容量组合 \(u_\ell=\alpha\tilde g_\ell+\beta\tilde{\bar s}_\ell\) 中 \(\alpha\)（与论文式 217 一致意图） |
| `--evo_beta_u` | `1.0` | 同上式中 \(\beta\) |
| `--expand_init_mode` | `zero` | 仅 `evorank`：`zero` 扩张时 B 列清零；`gradient` 按论文 Proposition 3.2 用 \(\partial L/\partial\Delta W\) 的投影近似之主奇异向量初始化新秩-1 槽位（多法同台时可统一写上，其它方法忽略） |

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
  --expand_init_mode gradient \
  --seed 42 \
  --log_dir runs/smoke \
  --output_dir artifacts \
  --export_csv results_smoke.csv
```

**B. 对比方法全开（较慢）**：在 A 的基础上加入 SoRA（gate + L1）。与终端冒烟一致时可为 EvoRank 打开 **`--expand_init_mode gradient`**（仅 `evorank` 生效，其余方法忽略）。

```bash
python run_benchmark.py \
  --methods lora adalora evorank sora toplora flatlora pissa \
  --task_name sst2 \
  --model_name roberta-base \
  --max_train_steps 50 \
  --T_es 20 \
  --warmup_ratio 0.1 \
  --adalora_delta_t 50 \
  --adalora_orth_reg_weight 0.1 \
  --sora_sparse_lambda 1e-3 \
  --sora_lambda_warmup_steps 0 \
  --flatlora_rho 0.05 \
  --expand_init_mode gradient \
  --seed 42 \
  --log_dir runs/smoke_full \
  --output_dir artifacts \
  --export_csv results_smoke_full.csv
```

**冒烟预期（健康检查）**：`--max_train_steps 50` 仅覆盖约一个 epoch 的一小段，**SST-2 上 `val_accuracy` 接近 0.5（随机基线）是正常现象**；多种方法数值相同或极接近也常见应确认：无 Python 异常、各方法均打印 `val_*` 行、**`[verify]`** 有输出、`artifacts/.../final/` 与 CSV 落盘。加载 `roberta-base` 时若出现 classifier 权重「新初始化」的提示，属在下游头随机初始化上的预期行为。

若使用 W&B，在以上命令末尾追加：`--use_wandb --wandb_project <your_project>`。

冒烟结束后可检查例如 `artifacts/sst2_roberta-base_lora/final/` 与 `metrics.jsonl`，并查看终端 **`[verify]`** 行是否与标签一致。

使用 nohup / 后台跑 torchrun 前建议：

```bash
mkdir -p logs
```

### 1.1) DDP 多卡冒烟测试（torchrun）

在 Linux 多卡上先用极小步数验证：各卡 loss/同步、EvoRank trial。

**重要说明（避免误读 `avg_rank=0`）**：

- **SoRA**：若在极短冒烟（例如 `--max_train_steps 50`）里使用论文级强稀疏（如 `--sora_sparse_lambda 10`），gate 很可能被快速压到全 0，从而打印 `gates=0 / avg_rank=0`。这通常是**超参导致的退化**，不代表代码崩溃。冒烟建议用温和系数（如 `1e-3`）并开启 `--sora_lambda_warmup_steps`。当前脚本不会再在短训练时隐式改写 `--sora_lambda_schedule`，若需要 `linear` 调度请显式传参。
- **AdaLoRA**：PEFT 中 `lora_E` 默认零初始化；在极短训练中，即便 `update_and_allocate` 正常触发，也可能长期显示 `eff_r=0`。本仓库优先从 `peft_config.rank_pattern`（布尔掩码求和）读取有效秩，回退口径才使用 `lora_E` 非零计数。出现 0 时请结合日志里的 `[diag] adalora_config` 一并判断，不建议仅凭单次冒烟结论判定算法失效。

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
  --expand_init_mode gradient \
  --seed 42 \
  --log_dir runs/ddp_smoke \
  --output_dir artifacts \
  --export_csv results_ddp_smoke.csv
```

若出现 `ProcessGroupNCCL::WorkNCCL::checkTimeout` / `ncclCommWatchdog`：多为某 rank **少做了一次集体通信**（历史上 EvoRank 在 `mutations` 列表长度不一致时会触发）。当前版本已在 `train_integration.py` 中对齐 trial 的 `all_reduce` 次数；若仍超时，可尝试增大 `export NCCL_TIMEOUT=1800` 或检查集群 IB/GPU 拓扑。

**对比全开（DDP 冒烟）**：在「最小三法」相同步数与 AdaLoRA 参数下，增加 `sora` 及对应 CLI。产物与 checkpoint **仅 rank0 写盘**。

```bash
torchrun --nproc_per_node=2 --master_port=29500 \
  run_benchmark.py \
  --ddp \
  --task_name sst2 \
  --model_name roberta-base \
  --methods lora adalora evorank sora toplora flatlora pissa \
  --max_train_steps 20 \
  --T_es 10 \
  --warmup_ratio 0.1 \
  --adalora_delta_t 10 \
  --adalora_orth_reg_weight 0.1 \
  --sora_sparse_lambda 1e-3 \
  --mini_val_k 4 \
  --flatlora_rho 0.05 \
  --expand_init_mode gradient \
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
> - **极小数据集易波动任务（需 10 ~ 20 Epochs）**：`mrpc`（以及 GLUE 的 `wnli` 这类极小任务）。非常容易出现大幅震荡或直接崩溃（如预测多数类别）。建议充分 Warmup 并放长训练轮数，最好配合 Early Stopping。
> - **大数据集易收敛的任务（3 ~ 5 Epochs 即可）**：`sst2`, `mnli`, `qnli`, `qqp`。这些数据量大的任务在 3~5 轮时通常已展现完美的收敛平台，为了最高精度最多尝试 5~10 轮。
> 如果你需要同时对所有子集跑 `--task_list ...`，建议为大规模和小规模数据集分别设置不同的训练脚本。

全对比时包含核心对比方法（lora adalora pissa evorank sora toplora flatlora）；若只想专注跑基线，可将 `--methods` 改为 `lora pissa adalora evorank` 并去掉 `--sora_sparse_lambda*` 等分支参数（**`--expand_init_mode` 仅影响 `evorank`，可保留 `gradient` 或改 `zero` 做消融**）。框架内部已经为 `sora`、`toplora`、`flatlora` 等强约束算法做好了动态超参兼容（如自动切换 `max_grad_norm=0.1` 和免衰减修正），因此**您可以在同一个命令中横向并跑这些算法**，而完全不需要手动切碎脚本调整那些苛刻的论文约束。
# === 脚本 A: 大数据集 (3~5 Epochs 即可收敛) ===
双卡（torchrun）+ nohup 后台运行：

```bash
nohup torchrun --nproc_per_node=2 --master_port=29500 \
  run_benchmark.py \
  --ddp \
  --methods lora adalora evorank sora toplora flatlora pissa \
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
  --sora_sparse_lambda 1e-3 \
  --lambda_c 0.0 \
  --complexity_mode rank_sum \
  --population_strategy all \
  --flatlora_rho 0.05 \
  --expand_init_mode gradient \
  --verify_n_samples 0 \
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
  --methods lora adalora evorank sora toplora flatlora pissa \
  --task_list cola mrpc stsb rte \
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
  --sora_sparse_lambda 1e-3 \
  --lambda_c 0.0 \
  --complexity_mode rank_sum \
  --population_strategy all \
  --flatlora_rho 0.05 \
  --expand_init_mode gradient \
  --verify_n_samples 0 \
  --seed 42 \
  --log_dir runs/main_results_small_ddp \
  --output_dir artifacts \
  --export_csv results_main_small_ddp.csv \
  > logs/main_results_small_ddp.out 2>&1 &
```

### 3) 消融模板（EvoRank）

**3a) ES / 复杂度等主消融（单配置）**：与论文 EvoRank 机制一致时，建议显式写上 **`--expand_init_mode gradient`**（扩张按 Proposition 3.2 初始化）；若对照「仅零初始化扩张」，改为 `zero`。

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
  --expand_init_mode gradient \
  --verify_n_samples 0 \
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
  --expand_init_mode gradient \
  --verify_n_samples 0 \
  --seed 42 \
  --log_dir runs/ablation_full_ddp \
  --output_dir artifacts \
  --export_csv results_ablation_full_ddp.csv \
  > logs/ablation_full_ddp.out 2>&1 &
```

**3b) 扩张初始化消融（`zero` vs `gradient`）**：除 `--expand_init_mode` 与输出路径/CSV 名外保持完全一致；合并进同一 CSV 时用不同 `--log_dir` 或后处理拼接。

```bash
# Baseline：扩张时 B 列清零（cold start）
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
  --expand_init_mode zero \
  --verify_n_samples 0 \
  --seed 42 \
  --log_dir runs/ablation/expand_zero \
  --output_dir artifacts \
  --export_csv results_ablation_expand_zero.csv

# 论文方向：投影梯度主奇异向量初始化（Proposition 3.2）
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
  --expand_init_mode gradient \
  --verify_n_samples 0 \
  --seed 42 \
  --log_dir runs/ablation/expand_gradient \
  --output_dir artifacts \
  --export_csv results_ablation_expand_gradient.csv
```

**3c) Reallocation 组合爆炸消融（`8` vs `0` 无限流）**：关闭限流对比 ES 评估耗时的退阶（在运行中能显著监控到 step 耗时的异常上升）。

```bash
# 论文方向：关闭 Reallocation 组合爆炸软约束（设为 0）
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
  --expand_init_mode gradient \
  --evo_max_reallocate_candidates 0 \
  --verify_n_samples 0 \
  --seed 42 \
  --log_dir runs/ablation/reallocate_unlimited \
  --output_dir artifacts \
  --export_csv results_ablation_reallocate_unlimited.csv
```

### 4) 效率模板

# === 脚本 A: 大数据集 ===
双卡（torchrun）+ nohup 后台运行：

```bash
nohup torchrun --nproc_per_node=2 --master_port=29500 \
  run_benchmark.py \
  --ddp \
  --methods lora adalora evorank sora toplora flatlora pissa \
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
  --sora_sparse_lambda 1e-3 \
  --lambda_c 0.001 \
  --complexity_mode rank_sum \
  --lambda_pop 16 \
  --population_strategy all \
  --expand_init_mode gradient \
  --verify_n_samples 0 \
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
  --methods lora adalora evorank sora toplora flatlora pissa \
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
  --sora_sparse_lambda 1e-3 \
  --expand_init_mode gradient \
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
  --methods lora adalora evorank sora toplora flatlora pissa \
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
  --sora_sparse_lambda 1e-3 \
  --expand_init_mode gradient \
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
  --expand_init_mode gradient \
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

## 公平对比规范（AdaLoRA / SoRA / EvoRank）

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

> `run_benchmark.py` 通过 `glue_metrics.py` 自动为每个任务选择对应的官方主指标（CSV 列 `val_metric_key`），无需手动指定。

### 统一参数来源

公平对比的统一参数从三个对比算法开源库中取**交集 / 最大公约数**：

| 参数 | 取值 | 来源说明 |
|------|------|----------|
| `model` | `microsoft/deberta-v3-base` | AdaLoRA / SoRA 均以此为主实验 backbone |
| `target_rank` | 8 | SoRA `lora_r=8` |
| `lora_alpha` | 16 | SoRA reproduce 用 10（及 alpha=16） |
| `target_modules` | 5 类 | AdaLoRA 6 类模块 PEFT 映射为 5 个 |
| `lr` | 8e-4（SoRA 论文表口径） | 当前 **`scripts/fair_glue_deberta_*.sh`** 按**任务**在各自脚本中设置（如 RTE 为 `1.2e-3`，见 `fair_glue_deberta_rte.sh`） |
| `batch_size` | 8（SoRA 论文表口径） | 当前公平管线默认 **`PER_DEVICE_BATCH_SIZE=32`**（`fair_glue_deberta_common.sh`），`torchrun` 进程数默认 ≤2，全局 batch 随 GPU 数变化 |
| `epochs` | 20（SoRA 论文表口径） | 各 GLUE 子任务在对应 `fair_glue_deberta_<task>.sh` 中单独指定（如 RTE `50`） |
| `warmup_ratio` | 0.06 | SoRA 默认 |
| `weight_decay` | 0.1 | SoRA 默认 |
| `max_grad_norm` | 0.1 | SoRA 默认 |
| `seed_list` | 0 21 42 81 100 | SoRA 官方 5 种子 |
| `sora_sparse_lambda` | 10 | SoRA `sparse_lambda=10` |
| `sora_sparse_lambda_2` | 3e-4 | SoRA `sparse_lambda_2=3e-4` |
| `toplora_dropout` | 0.05 | TopLoRA 默认 |
| `flatlora_rho` | 0.05 | Flat-LoRA 原文默认主线参数（见下文与论文扰动强度 σ 的对应关系） |

**Flat-LoRA 超参映射：** `--flatlora_rho` 对应 Flat-LoRA 论文中的扰动强度 σ。在视觉与 NLP 等任务上论文常用 **0.05**（主线）或 **0.10**（部分设定）；可按任务稳定性在二者间选择，并至少在表内注明所用 σ。

> [!IMPORTANT]
> **GLUE 公平脚本以「每任务一个 shell」为准**：`fair_glue_deberta.sh` 只负责批调度；**学习率 / epoch / max_length / lora_alpha / weight_decay / AdaLoRA 步级超参**均在对应的 **`scripts/fair_glue_deberta_<task>.sh`** 中写明，并由 **`fair_glue_deberta_common.sh`** 统一注入 `run_benchmark.py`。不要沿用本表「SoRA 论文表口径」列当作当前 shell 的单一默认值。

> [!NOTE]
> **制表**：`python scripts/generate_glue_table.py` 会读取根目录下 **`results_fair_glue_deberta_*.csv`**（每任务一个）；若你仍保留历史合并结果 `results_fair_glue_deberta_large_ddp.csv` / `small_ddp`，则脚本会优先使用这两个文件。

### 双协议与分表汇报（学术口径约定）

为同时满足「机制公平」与「论文复现」，建议将实验结果**分两张表**（或 CSV 分两批导出）汇报，并在论文/附录中写明所用协议：

| 协议 | 英文命名 | 定义 |
|------|----------|------|
| **协议 A** | *Controlled Fair Baseline* | **严格控制变量**：所有方法使用相同的 `target_modules`、相同的秩（`target_rank`）、相同的 `lr` / `batch_size` / 训练预算等；消除由层覆盖或容量不对齐带来的红利，**纯粹比较算法机制**。当前 `scripts/fair_*.sh` 横跑属于这一方向的工程实现。 |
| **协议 B** | *Author Defaults* | **忠实原著**：各方法允许采用原论文推荐的最优或默认超参组合，用于**复现论文宣称的性能**；与协议 A 的数字不宜混在同一主表直接比较，应分表并注明「论文协议」。 |

**Dropout 与双协议联动（建议）：**

- **协议 A**：对各方法的 LoRA / 适配器侧 **Dropout 全局统一**（推荐固定 **0.05**，与 TopLoRA 论文默认及多种 LoRA 变体常见取值一致；亦可在表内写死所选值）。
- **协议 B**：按各论文默认值透传（例如某方法原文用 0.1 则保留 0.1）。导出 CSV 时可用列 `effective_dropout` 核对实际注入值。

**模块覆盖策略（工程演进建议，与 `AUDIT_ALGO_FAIRNESS.md` 一致）：** 不宜仅为某一方法（如 SoRA）单独 hardcode「仅 Attention」。推荐在框架层提供统一入口，例如 `--target_modules` 与预设别名 `attn_only` / `all_linear`（或等价 `--module_preset`），再统一下发到全部对比方法，便于后续扩展新基线。

### `scripts/` 实验脚本与运行方式

以下脚本均在仓库根目录执行（`cd /path/to/EvoRLoRA`）。**Linux / macOS**：直接 `bash scripts/xxx.sh`。**Windows**：在 **Git Bash** 或 **WSL** 中运行 `.sh`；或在 PowerShell 中对照脚本内 `torchrun ...` 命令逐条执行。建议先 `mkdir -p logs artifacts`。

#### 脚本索引（`scripts/`）

| 脚本 | 场景 | 说明 |
|------|------|------|
| `fair_glue_deberta.sh` | GLUE 8 项批调度 | 依次分批启动 `fair_glue_deberta_<task>.sh`（含 **RTE**）；每任务 5 种子 × 7 方法；超参按任务脚本与 AdaLoRA NLU 对齐，见各 `fair_glue_deberta_*.sh` 头部注释 |
| `fair_glue_deberta_rte.sh` | 仅跑 RTE | 单任务入口；超参见脚本（当前为 `lr=1.2e-3`、`epochs=50` 等，与总批 `fair_glue_deberta.sh` 内 RTE 一致） |
| `fair_glue_deberta_*.sh` | GLUE 单任务 | `source fair_glue_deberta_common.sh` 后 `torchrun run_benchmark.py`，导出 **`results_fair_glue_deberta_<task>.csv`**，日志 **`logs/fair_glue_deberta_<task>.out`** |
| `fair_nlg_xsum.sh` | XSum 摘要（NLG） | BART-large，对齐 AdaLoRA Table 3 协议；`run_benchmark.py --task_type nlg` |
| `fair_nlg_cnndailymail.sh` | CNN/DailyMail 摘要（NLG） | 同上 |
| `generate_nlg_table.py` | NLG 结果汇总 | 从 fair NLG 导出的 CSV 生成 **AdaLoRA Table 3 风格** ROUGE 表 |
| `fair_qa_squadv1.sh` | SQuAD v1.1 抽取式 QA | `run_qa_benchmark.py`：DeBERTa-v3-base × 4 档 rank × 多方法，对齐 AdaLoRA **Table 2** |
| `fair_qa_squadv2.sh` | SQuAD v2.0 抽取式 QA | 同上，含不可答样本 |
| `generate_qa_table.py` | QA 结果汇总 | 从 `--export_csv` 或 `eval_results.json` 生成 **AdaLoRA Table 2 风格** EM/F1 表 |
| `nlg_pissa_benchmark.sh` | Causal LM SFT + 下游评测 | `run_nlg_benchmark.py` + `eval_nlg_pissa.py`，对齐 PiSSA **Table 2**（GSM8K/MATH/HumanEval/MBPP 等） |
| `eval_nlg_pissa.py` / `summarize_nlg_pissa.py` | PiSSA 评测与制表 | 由 `nlg_pissa_benchmark.sh` 调用；也可单独对已有 checkpoint 跑评测 |
| `ablate_evorank_glue_deberta.sh` | EvoRank 主消融 | Full / no-complexity / zero-init / no-EMA / no-persistence-cooldown / no-reallocation / no-noop |
| `ablate_evorank_reallocation_efficiency.sh` | Reallocation 效率消融 | 对比 `K_realloc=8` 与 unlimited |
| `generate_glue_table.py` | GLUE 主表 Markdown | 自动读取根目录下各 **`results_fair_glue_deberta_<task>.csv`**（8 个任务）；若仍存在历史合并文件 `results_fair_glue_deberta_{large,small}_ddp.csv` 则优先用后者 |

#### 1) GLUE（DeBERTa 公平对比）

调度入口 **`scripts/fair_glue_deberta.sh`** 会按批（默认 `PARALLEL_JOBS=2`）启动 8 个子脚本：`cola`、`mnli`、`mrpc`、`qqp`、`qnli`、`rte`、`sst2`、`stsb`。每个子脚本通过 **`scripts/fair_glue_deberta_common.sh`** 调用 `torchrun run_benchmark.py`，在仓库根目录追加写入：

- `results_fair_glue_deberta_<task>.csv`（例如 `results_fair_glue_deberta_sst2.csv`）
- 日志：`logs/fair_glue_deberta_<task>.out`

```bash
# 8 项 GLUE（含 RTE），前台或后台均可
mkdir -p logs
bash scripts/fair_glue_deberta.sh

# 仅重跑 RTE（与总批中的 RTE 配置相同）
bash scripts/fair_glue_deberta_rte.sh

# 仅重跑某一任务示例
bash scripts/fair_glue_deberta_sst2.sh

# 制表：自动聚合上述 per-task CSV（可选写入 Markdown）
python scripts/generate_glue_table.py --out_md artifacts/glue_table.md
```

#### 2) NLG 摘要 XSum / CNN-DailyMail（AdaLoRA Table 3 协议）

```bash
nohup bash scripts/fair_nlg_xsum.sh > logs/fair_nlg_xsum_ddp.out 2>&1 &
nohup bash scripts/fair_nlg_cnndailymail.sh > logs/fair_nlg_cnndm_ddp.out 2>&1 &

# 跑完后由 CSV 生成 ROUGE 汇总表
python scripts/generate_nlg_table.py \
  --xsum_csv results_fair_nlg_xsum_ddp.csv \
  --cnndm_csv results_fair_nlg_cnndm_ddp.csv \
  --out_md artifacts/nlg/table3_nlg_rouge.md
```

#### 3) 抽取式 QA：SQuAD v1.1 / v2.0（AdaLoRA Table 2 协议）

入口为根目录 **`run_qa_benchmark.py`**（非 `run_benchmark.py`）。公平脚本默认 **2 卡** `torchrun`，可通过环境变量覆盖：

| 变量 | 默认 | 含义 |
|------|------|------|
| `MODEL` | `microsoft/deberta-v3-base` | 骨干 |
| `RANKS` | `1 2 4 8` | 四档预算（对应论文约 0.08% / 0.16% / 0.32% / 0.65%） |
| `METHODS` | `lora pissa adalora evorank sora flatlora toplora` | 方法列表（空格分隔） |
| `SEED` | `42` | 随机种子 |
| `EPOCHS` | `3` | 训练 epoch（论文完整复现可改为 `10` 等） |
| `CSV` | `results_fair_qa_squadv1.csv` / `...squadv2.csv` | 追加写入的汇总 CSV |
| `NPROC` | `2` | `torchrun --nproc_per_node` |
| `MAX_TRAIN_SAMPLES` / `MAX_EVAL_SAMPLES` | `0` | 非 0 时仅用于调试子集 |
| `MAX_TRAIN_STEPS` | `0` | 非 0 时覆盖 `EPOCHS`（冒烟用） |

```bash
# SQuAD v1.1
bash scripts/fair_qa_squadv1.sh

# SQuAD v2.0
bash scripts/fair_qa_squadv2.sh

# 冒烟示例（单进程、小数据、50 step）
conda activate ./envs/evorank   # 或你的 conda 环境
python run_qa_benchmark.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --method lora --dataset_name squad \
  --max_train_steps 50 --max_train_samples 400 --max_eval_samples 100 \
  --bf16 --output_dir artifacts/qa/smoke --export_csv results_qa_smoke.csv --seed 42

# 制表
python scripts/generate_qa_table.py --csv results_fair_qa_squadv1.csv --task squad \
  --out_md artifacts/qa/table2_squadv1.md
python scripts/generate_qa_table.py --csv results_fair_qa_squadv2.csv --task squad_v2 \
  --out_md artifacts/qa/table2_squadv2.md
```

#### 4) PiSSA 风格 Causal LM（Table 2：SFT + 下游生成评测）

```bash
# 编辑 scripts/nlg_pissa_benchmark.sh 内 MODEL / TRAIN_SUB_TASK 后执行
bash scripts/nlg_pissa_benchmark.sh

# 仅汇总已有 artifacts/nlg/**/eval_results.json
python scripts/summarize_nlg_pissa.py --root artifacts/nlg --out_md artifacts/nlg/table2_pissa.md
```

调试时可设环境变量：`EVAL_MAX_SAMPLES=200`、`USE_VLLM=1`（需已安装 vLLM）。

#### 5) EvoRank 消融

后台跑消融示例：

```bash
nohup bash scripts/ablate_evorank_glue_deberta.sh > logs/ablate_evorank_glue_deberta.out 2>&1 &
nohup bash scripts/ablate_evorank_reallocation_efficiency.sh > logs/ablate_evorank_realloc.out 2>&1 &
```

公平脚本均使用 `--methods lora adalora evorank sora toplora flatlora pissa` 一次性横跑全部方法，保证控制变量公平。EvoRank 的 `--evorank_r_max` / `--evo_alpha_u` / `--evo_beta_u` 未写入 shell 时与 `run_benchmark.py` 默认值一致（16 / 1.0 / 1.0）；**公平脚本已统一追加 `--expand_init_mode gradient`**（仅 EvoRank 使用，对齐论文 Proposition 3.2 的扩张初始化；消融 `zero` 时请自行改参）。若消融容量统计或提高扩张 ceiling，再在命令中显式追加即可。

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
- `glue_metrics.py`：GLUE 各子集验证主指标（Matthews / Acc / F1 / Pearson-Spearman 均值等）
- `run_benchmark.py`：GLUE / NLG 主实验入口（`--task_type nlu|nlg`）
- `run_nlg_benchmark.py`：Causal LM 指令微调（PiSSA 数据集 / 多方法 PEFT）
- `run_qa_benchmark.py`：抽取式 QA（SQuAD v1.1 / v2.0，EM/F1，`--export_csv` 汇总）
- `scripts/fair_glue_deberta.sh`：GLUE 8 任务批调度（调用各 `fair_glue_deberta_<task>.sh`，含 RTE；每任务导出 `results_fair_glue_deberta_<task>.csv`）
- `scripts/fair_glue_deberta_rte.sh` / `fair_glue_deberta_sst2.sh` 等：单任务公平对比（共享 `fair_glue_deberta_common.sh`）
- `scripts/fair_nlg_xsum.sh`：XSum 全方法横向公平对比（含 `--expand_init_mode gradient`）
- `scripts/fair_nlg_cnndailymail.sh`：CNN/DailyMail 全方法横向公平对比（含 `--expand_init_mode gradient`）
- `scripts/generate_nlg_table.py`：NLG ROUGE 结果 → AdaLoRA Table 3 风格 Markdown
- `scripts/fair_qa_squadv1.sh` / `scripts/fair_qa_squadv2.sh`：SQuAD QA 公平 sweep（见上文「`scripts/` 实验脚本与运行方式」）
- `scripts/generate_qa_table.py`：QA EM/F1 → AdaLoRA Table 2 风格 Markdown
- `scripts/nlg_pissa_benchmark.sh`：PiSSA Table 2 风格训练 + 下游评测编排
- `scripts/eval_nlg_pissa.py` / `scripts/summarize_nlg_pissa.py`：PiSSA 评测与汇总
- `scripts/ablate_evorank_glue_deberta.sh`：EvoRank 主消融（精度侧创新验证）
- `scripts/ablate_evorank_reallocation_efficiency.sh`：Reallocation 组合爆炸防护效率消融
- `scripts/generate_glue_table.py`：由 fair GLUE CSV 生成 Markdown 对比表
- `scripts/summarize_evorank_ablation.py`：汇总主消融结果，生成 `results_evorank_ablation_summary.csv/.md`
- `scripts/summarize_evorank_reallocation_efficiency.py`：汇总 Reallocation 效率消融结果，生成 `results_evorank_reallocation_efficiency_summary.csv/.md`

主消融跑完后可直接执行：

```bash
python scripts/summarize_evorank_ablation.py
python scripts/summarize_evorank_reallocation_efficiency.py
```

会在仓库根目录下生成：

- `results_evorank_ablation_summary.csv`
- `results_evorank_ablation_summary.md`
- `results_evorank_reallocation_efficiency_summary.csv`
- `results_evorank_reallocation_efficiency_summary.md`
