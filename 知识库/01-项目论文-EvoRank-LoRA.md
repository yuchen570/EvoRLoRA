---
tags: [论文, EvoRank-LoRA, LoRA, ES]
status: draft
updated: 2026-04-05
---

# EvoRank-LoRA（项目主论文）

## 元信息
- 标题: `EvoRank-LoRA: Evolutionary Bidirectional Rank Adaptation for Low-Rank Fine-Tuning`
- 来源文件: [evo_rank_lora_neurips_with_thresholds.tex](../论文/evo_rank_lora_neurips_with_thresholds.tex), [mian.tex](../论文/mian.tex)

## 一句话总结
- 将 LoRA 的秩分配建模为离散结构优化问题，采用 ES 动态执行扩秩/减秩/跨层重分配，与梯度训练交替进行。

## 核心机制
1. 在每个 LoRA 层定义最大秩超空间（`R_max`）。
2. 用二值掩码激活 rank-1 分量，得到有效秩 `r_l`。
3. 内循环: 固定结构，更新 LoRA 连续参数。
4. 外循环: 基于验证集奖励，ES 采样结构变异并精英选择。

## 结构演化算子
- Expansion（扩秩）: 在高需求层激活新的 rank-1 分量。
- Reduction（减秩）: 在低重要性分量上做结构裁剪。
- Reallocation（重分配）: 在近似固定总预算下跨层迁移容量。

## 关键公式（摘录）
- `ΔW_l = Σ_{i=1..R_max} m_{l,i} b_{l,i} a_{l,i}^T`
- 结构目标: `J(z) = L_val(Θ; z) + λ_c C(z)`
- 奖励: `R(z') = -L_val(Θ; z') - λ_c C(z')`

## 动态阈值实现点
- 对层级需求分数与分量重要性分数做归一化 + EMA 平滑。
- 用分位数阈值触发扩秩（上分位）和减秩（下分位）。
- 稳定机制: warmup、持续触发计数、cooldown 防抖。
- 参考配置: `p_g=80%`, `p_p=10%`, `rho=0.9`, `r_min=2`, `r_max=16`。

## 理论主张（待实验验证）
- 扩秩扩大可行更新空间。
- 减秩在平滑假设下有局部损失上界。
- 精英 ES 在结构选择阶段单调改进正则化目标。
- 双时间尺度交替优化收敛到块局部最优。
- 补充点: 外层结构会在有限步内稳定，之后退化为固定结构下的常规 LoRA 优化。

## 实验设计细化
- 核心问题:
  - 与固定秩 LoRA 在同预算下是否更优。
  - 与 AdaLoRA 相比是否有更好性能-效率折中。
  - 扩秩/减秩是否都必要。
  - ES 是否优于随机/贪心结构搜索。
- 指标建议:
  - 任务指标: Acc/F1/Rouge/BLEU/PPL。
  - 效率指标: 训练参数量、显存峰值、训练耗时、有效秩均值。
  - 结构指标: 每层秩轨迹、总结构变更次数。

## 与基线的关系
- 对 [[02-论文-LoRA]]: 从固定秩改为动态秩。
- 对 [[03-论文-AdaLoRA]]: 强调离散进化结构更新。
- 对 [[04-论文-SoRA]]: 重点不同（动态结构 vs 稀疏低秩）。
- 对 [[05-论文-LoRA-GA]]: 互补（初始化/梯度近似 vs 结构进化）。

## 待补条目
- 任务设置与数据集
- 与现有实现代码的精确映射（模块、函数、超参）
- 结果表格与消融结论

## 权威来源
- arXiv (LoRA): https://arxiv.org/abs/2106.09685
- OpenReview (AdaLoRA): https://openreview.net/forum?id=lq62uWRJjiY
- arXiv (LoRA-GA): https://arxiv.org/abs/2407.05000
- arXiv (ES 2017): https://arxiv.org/abs/1703.03864
- arXiv (ES Hyperscale): https://arxiv.org/abs/2511.16652


## 三件套卡片
- [[cards/01-A-公式卡-EvoRank-LoRA.md|公式卡]]
- [[cards/01-B-复现实验卡-EvoRank-LoRA.md|复现实验卡]]
- [[cards/01-C-失败案例卡-EvoRank-LoRA.md|失败案例卡]]
