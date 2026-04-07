# 对比算法一致性与公平性审查报告

## 1) 统一公平比较协议（已落地定义）

本报告采用以下统一口径审查是否“公平”：

- 同任务与同数据切分：同一 `task_name/task_list`、同一 `task_type`。
- 同预算：`epochs/max_train_steps`、`batch_size`、`max_length`、`seed/seed_list` 一致。
- 同优化主框架：`lr`、`warmup_ratio`、`weight_decay`、`max_grad_norm`、scheduler 一致。
- 方法特有参数仅允许影响该方法内部机制（如 AdaLoRA 的 rank allocator，SoRA 的稀疏门控），不得引入额外训练阶段或明显更多预算。
- 容量口径双报告：同时报告 `target_rank` 与“实际可训练参数量（trainable params）”。

---

## 2) 逐算法一致性审计

## AdaLoRA

- 结论：**核心机制基本忠实**，实现依赖 PEFT（非仓库自写 AdaLoRA 本体）。
- 证据：
  - `run_benchmark.py` 中使用 `AdaLoraConfig`，传入 `init_r/target_r/tinit/tfinal/deltaT`。
  - `adalora_utils.py` 显式补充正交正则，并在每步后执行 `update_and_allocate`。
- 风险：
  - 与论文逐行一致性依赖 `peft>=0.18.1` 具体实现版本。
  - DeBERTa 下 AdaLoRA 固定 `lora_dropout=0.1`，与 LoRA/PiSSA 分支（0.0）不一致。

## SoRA

- 结论：**核心思想忠实（gate + L1 + 近端阈值）**，但默认覆盖层更广、训练机制与其他方法异构。
- 证据：
  - `SoRA/src/lora.py`：`gate` 参数、前向 `(... @ A^T).mul(gate) @ B^T`。
  - `SoRA/src/trainer.py`：损失增加 `|gate|` 稀疏项。
  - `SoRA/src/sparse_optimizer.py`：软阈值/近端收缩更新。
- 风险：
  - 默认修改模块包括 attention + FFN（`q/v/k/proj/ff.w1/ff.w2`），可能高于论文或基线常用的窄覆盖。
  - `config/lora_config.json` 解冻了 `layer_norm/final_layer_norm`，不再是“纯 LoRA-only”参数集。

## PiSSA

- 结论：**以 PEFT 的 PiSSA 初始化路径实现，工程上可用但非仓库内置论文实现**。
- 证据：
  - `run_benchmark.py` 中 `method_name=="pissa"` 仅设置 `init_lora_weights="pissa"` 后走 `get_peft_model`。
- 风险：
  - 论文级 SVD/主子空间细节完全依赖 PEFT 版本，不可在本仓库单独审计到位。

## TopLoRA

- 结论：**实现了论文主公式路径（token-wise λ）**，但与 LoRA/PiSSA 同 rank 时参数量并不等价。
- 证据：
  - `toplora_inject.py`：`lambda(x)=exp(RMSNorm(x@W_lambda))`，并在 LoRA 中按 token 逐维缩放。
  - 注入后除基座冻结外，解冻分类头参数。
- 风险：
  - 每层新增 `W_lambda`（`d_in x r`）等参数，同 rank 条件下可训练参数量通常高于 LoRA/PiSSA，需显式披露。

## Flat-LoRA

- 结论：**当前实现与论文核心机制基本一致**（全参数空间随机扰动、filter-wise 强度、按宽度缩放、cosine-increasing 调度、seed+小统计量低内存还原），且与 LoRA 共享同一可训练参数集，具备较好的公平可比性。
- 说明：
  - 仓库内未发现独立目录 `D:/EvoRLoRA/Flat-LoRA`，当前实现位于 `flatlora_inject.py`，训练接入在 `run_benchmark.py`。
  - 论文依据已对照 `对比实验论文/flatlora.pdf`（Eq.(7) 与 cosine-increasing 策略）。
- 证据：
  - `flatlora_inject.py` 在 LoRA 层前向前钩子中构造 `W' = W + sBA` 后注入高斯扰动，并在反向钩子中按相同 seed 还原权重。
  - 扰动强度采用 `factor=0.5*(1-cos(pi*t/T))` 的随步增长调度，且按输入维度 `1/sqrt(n)` 缩放，匹配论文“宽度无关方差”思路。
  - `run_benchmark.py` 中 `method_name=="flatlora"` 走 LoRA 注入分支（与 LoRA/PiSSA 同参数训练），再附加 `FlatLoRAHookManager`。
- 风险：
  - 论文超参名通常记为 `sigma`，仓库参数名为 `flatlora_rho`，语义一致但命名不同，文档需明确映射。
  - `flatlora_inject.py` 注释里的分布写法与代码实现的“标准差/方差”表述不够严格一致，建议补充注释避免歧义（不影响实际逻辑）。
  - Flat-LoRA 相比 LoRA 会引入额外随机扰动过程（轻微时间开销），报告中应标注“同参数量、不同训练扰动策略”。

---

## 3) 一致性与公平性问题矩阵（带证据）

| 优先级 | 问题 | 影响 | 证据文件 |
|---|---|---|---|
| 高 | README 与脚本超参不一致（如 GLUE lr、RTE 预算） | 结果解读与复现口径混乱 | `README.md`、`scripts/fair_glue_deberta.sh`、`scripts/fair_glue_deberta_rte.sh` |
| 高 | `seed_list` 下数据加载在种子循环外，且 `setup_data_and_model(seed=seeds[0])` 固定首种子 | 多 seed 实验的采样随机性不完全独立 | `run_benchmark.py` |
| 高 | TopLoRA 与 LoRA/PiSSA 在同 rank 下参数量不等 | “同 rank 公平”表述存在偏差 | `toplora_inject.py`、`run_benchmark.py` |
| 中 | SoRA 默认模块覆盖含 FFN，且解冻 LayerNorm | 与窄覆盖 LoRA 基线对比可能偏移 | `SoRA/src/lora.py`、`SoRA/config/lora_config.json` |
| 中 | AdaLoRA/SoRA 与 LoRA/PiSSA 默认 dropout 不一致 | 比较中混入额外正则差异 | `run_benchmark.py` |
| 中 | Flat-LoRA 参数命名与论文符号存在映射差异（`rho` vs `sigma`） | 复现实验时可能误配扰动强度 | `flatlora_inject.py`、`run_benchmark.py` |
| 中 | SoRA 存在方法特有双优化器与阈值逻辑 | 训练机制不可完全同构（需在论文中说明） | `run_benchmark.py`、`SoRA/src/sparse_optimizer.py` |
| 低 | 部分调试日志仅覆盖 lora/evorank/pissa | 可观测性不对称，但不改训练数学 | `run_benchmark.py` |

---

## 4) 修复完成状态（逐项勾选）

状态说明：`[x]` 已完成，`[ ]` 未完成。

## 必须修复（建议先做）

- [x] 统一并冻结“主结果协议”，避免混用不同脚本口径。  
  - 已完成：`README.md` 中 `fair_glue_deberta.sh` / `fair_glue_deberta_rte.sh` 的 `lr/epochs/batch_size` 已与当前脚本实参对齐（8e-4 / 20 / 8）。
- [x] 修复多 seed 数据随机性口径。  
  - 已完成：`run_benchmark.py` 在 `for seed in seeds` 内重建 DataLoader（`make_loaders(seed)`），采样随机性随 seed 独立。
- [x] 在结果表中增加容量字段。  
  - 已完成：CSV 已包含 `target_rank` 与 `trainable_params`；并保留 `extra_params` 用于披露 TopLoRA 额外容量。

## 建议修复（已落实）

- [x] 提供“两套并行协议”并分表汇报（建议写入设计文档并在 README 固化定义）。  
  - `Protocol A: ControlledFairBaseline`（严格控制变量）：所有方法强制使用相同 `target_modules`、相同 `rank`、相同 `lr/batch_size`，尽可能消除架构红利，仅比较算法机制。  
  - `Protocol B: AuthorDefaults`（忠实原著）：允许各方法使用论文推荐超参，目标是复现论文宣称性能上限。  
- [x] 统一 Dropout 策略并与双协议联动，避免隐式差异。  
  - 在 `Protocol A` 中建议全局固定（默认可设 `0.05`）。  
  - 在 `Protocol B` 中按各论文默认值透传。  
- [x] 将“仅 Attention 层”从方法特例升级为全局模块策略（避免 SoRA-only hardcode）。  
  - 建议新增统一入口：`--module_preset attn_only|all_linear|custom`（或保持 `--target_modules` + 预设别名），并由框架底层统一下发到所有方法。  
- [x] 在 README 中显式声明 Flat-LoRA 映射关系与推荐范围。  
  - 建议文案：`flatlora_rho` 对应论文扰动强度 `sigma`，推荐默认范围 `0.05` 或 `0.10`（按任务规模与稳定性选择）。

## 可选修复

- [x] 扩展日志覆盖到 `toplora/sora/adalora`，便于问题定位。  
  - 已完成：`run_benchmark.py` 的优化器参数日志覆盖已扩展至 `adalora/sora/toplora`。
- [x] 在导出 CSV 中增加 `target_modules`、`effective_dropout`、`optimizer_type` 字段。  
  - 已完成：`run_benchmark.py` CSV `fieldnames` 与 `res.update(...)` 均已写入上述字段。

---

## 5) 统一运行模板（建议作为主协议）

以下模板强调“同预算+同层覆盖+同评估口径”，用于横向主表：

```bash
torchrun --nproc_per_node=2 run_benchmark.py \
  --ddp \
  --methods lora adalora evorank sora toplora flatlora pissa \
  --task_list mnli sst2 cola qqp qnli mrpc stsb rte \
  --model_list microsoft/deberta-v3-base \
  --target_rank 8 \
  --lora_alpha 16 \
  --epochs 20 \
  --batch_size 8 \
  --max_length 128 \
  --lr 8e-4 \
  --warmup_ratio 0.06 \
  --weight_decay 0.1 \
  --max_grad_norm 0.1 \
  --seed_list 0 21 42 81 100 \
  --adalora_delta_t 100 \
  --adalora_orth_reg_weight 0.1 \
  --sora_sparse_lambda 10 \
  --sora_sparse_lambda_2 3e-4 \
  --flatlora_rho 0.05 \
  --export_csv results_fair_main.csv
```

说明：
- 若强调“方法忠实”，可保留各方法专属默认；但必须在表格中增加“非同构项”列（如 SoRA 双优化器、TopLoRA 额外参数）。
- 若强调“严格同构公平”，需额外统一覆盖层、dropout 与解冻策略。

---

## 6) 最终结论

- 论文一致性：  
  - AdaLoRA、SoRA、TopLoRA、Flat-LoRA 的核心机制在当前仓库均可识别且基本成立；PiSSA 为 PEFT 托管实现。  
- 公平性：  
  - 当前仓库已具备“统一横跑”基础，但仍存在 **高优先级公平风险**（协议混用、多种子采样口径、同 rank 不同容量未披露）；Flat-LoRA 本身参数量公平性较好，但需补充 `rho/sigma` 映射说明。  
- 建议：  
  - 在后续论文/报告中，至少按本报告第 4 节完成“必须修复”三项后再发布最终公平结论。
