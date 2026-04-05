---
tags: [paper-learning, 论文学习, EvoRLoRA, 文献综述]
status: active
updated: 2026-04-05
---

# Paper Learning 总笔记（EvoRLoRA 方向）

## 学习目标
- 围绕 `LoRA -> 动态秩 -> 稀疏低秩 -> 进化策略` 建立一套可扩展知识库。
- 直接服务于 [[01-项目论文-EvoRank-LoRA]] 的实验与写作。

## 模块一：论文背景

### 研究背景
- PEFT（参数高效微调）核心矛盾：在接近全参数微调效果的同时，尽量减少可训练参数、显存和存储开销。
- LoRA 路线的核心问题：固定秩通常无法匹配“不同层、不同训练阶段”的容量需求差异。
- EvoRLoRA 的关键定位：将秩分配显式建模为离散结构优化问题，用 ES 做外层结构更新。
- 方法演进脉络（本库重点）：
  - `LoRA`：固定秩低秩增量。
  - `AdaLoRA`：预算分配自适应。
  - `SoRA`：稀疏+低秩联合。
  - `LoRA-GA`：训练策略增强（梯度近似）。
  - `EvoRank-LoRA`：离散结构演化（扩秩/减秩/重分配）。

### 补充资源
| 类型 | 标题 | 链接 | 日期 | 关联说明 |
|---|---|---|---|---|
| 文档 | Hugging Face PEFT 文档首页 | https://huggingface.co/docs/peft/index | 2026-04-05 | 提供 PEFT 全景，帮助统一术语与方法谱系。 |
| 论文 | LoRA (arXiv:2106.09685) | https://arxiv.org/abs/2106.09685 | 2026-04-05 | LoRA 原始定义与效率收益来源。 |
| 论文 | AdaLoRA (ICLR 2023 OpenReview) | https://openreview.net/forum?id=lq62uWRJjiY | 2026-04-05 | 动态预算分配代表工作，对比 EvoRLoRA 的重要基线。 |
| 文档 | PEFT LoRA Developer Guide | https://huggingface.co/docs/peft/main/en/developer_guides/lora | 2026-04-05 | 提供工程实现视角，便于落地实验。 |

## 模块二：论文内容

### 2A 问题定义与动机
- 问题：固定 LoRA rank 在层间和时间维度上都可能失配。
- 动机：结构变量（开/关 rank-1 分量）本质离散，单纯梯度法不自然。
- 任务视角：
  - 层间异质性：注意力层、FFN 层的适配需求不同。
  - 时序异质性：训练前期更需探索容量，后期更需压缩冗余。
- 失败模式（如果不做动态结构）：
  - 关键层欠配导致性能瓶颈。
  - 非关键层过配导致参数浪费和泛化不稳。

### 2B 方法与核心创新
- 超空间参数化：每层预留 `R_max`，通过二值掩码得到有效秩。
- 交替优化：内环更新连续权重，外环 ES 变异结构（扩秩/减秩/重分配）。
- 选择机制：基于验证奖励的精英保留，抑制结构抖动。
- 候选评分核心：
  - 奖励函数兼顾验证损失与复杂度惩罚。
  - 结构仅在候选优于当前方案时更新，避免无效扰动。
- 相比 AdaLoRA 的关键差别：
  - AdaLoRA 更偏预算连续调配。
  - EvoRank-LoRA 更强调显式离散结构搜索与局部变异。

### 2C 技术细节
- 扩秩信号：层级需求分数（如梯度范数相关）与重要性统计结合。
- 减秩信号：按分量贡献/重要性打分并做候选裁剪。
- 动态阈值：分位数 + EMA 平滑，提升跨阶段稳定性。
- 稳定策略（来自项目 tex）：
  - Warmup：前期不做结构变更。
  - Persistence：信号连续满足若干次才执行变更。
  - Cooldown：新扩张分量短期内不允许立刻被剪掉。
- 默认示例超参（稿件）：
  - `p_g=80%`, `p_p=10%`, `rho=0.9`, `r_min=2`, `r_max=16`。
  - 结构更新间隔 `T_es=200` step，warmup 覆盖前 `10%` 训练。

### 2D 实验与结果（知识库模板）
- 任务维度：GLUE / QA / NLG（可按你代码仓具体任务更新）。
- 对比维度：性能、训练时长、显存、参数量、推理开销。
- 消融维度：仅扩秩、仅减秩、重分配关闭、阈值策略替换。
- 结果记录建议（后续填表）：
  - 主指标：任务分数（Acc/F1/Rouge 等）。
  - 资源指标：训练参数量、峰值显存、吞吐、总训练时长。
  - 结构指标：各层有效秩轨迹、最终秩分布、变更次数。
- 公平对比约束：
  - 相同 backbone 与数据切分。
  - 对齐训练总步数与优化器配置。
  - 报告至少 3 次随机种子均值与方差。

### 2E 结论与局限
- 结论预期：动态秩应在低预算下优于固定秩基线。
- 潜在局限：外层搜索额外验证开销、阈值与变异策略敏感性。
- 推进方向：
  - 把结构搜索从启发式升级为更强代理目标或贝叶斯优化。
  - 研究任务迁移下的“秩先验”复用，减少冷启动成本。
  - 融合稀疏化与动态秩，进一步压缩训练与推理成本。

### 模块二补充资源
| 类型 | 标题 | 链接 | 日期 | 关联说明 |
|---|---|---|---|---|
| 论文 | LoRA (arXiv) | https://arxiv.org/abs/2106.09685 | 2026-04-05 | 低秩增量建模的起点。 |
| 论文 | AdaLoRA (OpenReview) | https://openreview.net/forum?id=lq62uWRJjiY | 2026-04-05 | 动态预算分配的重要对照。 |
| 论文 | SoRA (ACL Anthology) | https://aclanthology.org/2023.emnlp-main.252/ | 2026-04-05 | 稀疏低秩路线，与动态秩互补。 |
| 论文 | LoRA-GA (arXiv) | https://arxiv.org/abs/2407.05000 | 2026-04-05 | 训练效率增强路线，可与结构演化结合。 |

## 模块三：基础知识索引
- [[12-基础知识-PEFT与LoRA生态]]
- [[13-基础知识-低秩分解与秩动态化]]
- [[14-基础知识-进化策略ES与结构搜索]]

### 模块三补充资源
| 类型 | 标题 | 链接 | 日期 | 关联说明 |
|---|---|---|---|---|
| 论文 | Evolution Strategies (2017) | https://arxiv.org/abs/1703.03864 | 2026-04-05 | ES 的经典可扩展性论证。 |
| 论文 | Evolution Strategies at the Hyperscale | https://arxiv.org/abs/2511.16652 | 2026-04-05 | 大模型规模下 ES 的效率分析。 |
| 文档 | JSON Canvas Spec 1.0 | https://jsoncanvas.org/spec/1.0/ | 2026-04-05 | 图谱结构的规范依据。 |

## 模块四：综合检验（自测）
- 用 3 句话解释 EvoRLoRA：问题、方法、收益。
- 给出一个“何时不该扩秩”的判据示例。
- 说明 ES 外环与梯度内环的职责边界。
- 给出一个你当前实验中“结构更新最频繁层”的解释假设。
- 比较 `AdaLoRA` 与 `EvoRank-LoRA` 在“搜索空间表达力”上的差异。

## 论文卡片入口
- [[01-项目论文-EvoRank-LoRA]]
- [[02-论文-LoRA]]
- [[03-论文-AdaLoRA]]
- [[04-论文-SoRA]]
- [[05-论文-LoRA-GA]]
- [[06-论文-Evolution-Strategies-2017]]
- [[07-论文-Evolution-Strategies-Hyperscale]]
- [[40-论文对比矩阵-EvoRLoRA]]


## 三件套卡片索引
- [[cards/01-A-公式卡-EvoRank-LoRA]] | [[cards/01-B-复现实验卡-EvoRank-LoRA]] | [[cards/01-C-失败案例卡-EvoRank-LoRA]]
- [[cards/02-A-公式卡-LoRA]] | [[cards/02-B-复现实验卡-LoRA]] | [[cards/02-C-失败案例卡-LoRA]]
- [[cards/03-A-公式卡-AdaLoRA]] | [[cards/03-B-复现实验卡-AdaLoRA]] | [[cards/03-C-失败案例卡-AdaLoRA]]
- [[cards/04-A-公式卡-SoRA]] | [[cards/04-B-复现实验卡-SoRA]] | [[cards/04-C-失败案例卡-SoRA]]
- [[cards/05-A-公式卡-LoRA-GA]] | [[cards/05-B-复现实验卡-LoRA-GA]] | [[cards/05-C-失败案例卡-LoRA-GA]]
- [[cards/06-A-公式卡-ES-2017]] | [[cards/06-B-复现实验卡-ES-2017]] | [[cards/06-C-失败案例卡-ES-2017]]
- [[cards/07-A-公式卡-ES-Hyperscale]] | [[cards/07-B-复现实验卡-ES-Hyperscale]] | [[cards/07-C-失败案例卡-ES-Hyperscale]]
