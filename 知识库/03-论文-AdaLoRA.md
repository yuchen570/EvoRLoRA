---
tags: [论文, AdaLoRA, PEFT, LoRA]
status: active
updated: 2026-04-05
---

# AdaLoRA: Adaptive Budget Allocation for PEFT

## 原文
- [ADALORA.pdf](../论文/ADALORA.pdf)

## 核心观点
- 在总预算约束下做层间秩预算分配，比固定秩更高效。
- 关注预算自适应分配与性能平衡。

## 2A 问题定义与动机
- 固定 rank LoRA 对所有层一刀切，容量利用不均。
- 目标是在有限参数预算内把容量分配到更关键层。

## 2B 方法与创新
- 通过重要性估计实现动态预算分配。
- 训练过程中持续调整各层 rank/预算，提升参数效率。

## 2C 技术细节
- 关键在于“重要性评估 + 预算调度”的稳定性。
- 对训练阶段差异敏感，通常需要合理 warmup 与调度策略。

## 2D 实验观察
- 相比固定秩 LoRA，常在同预算下取得更好结果。
- 在低预算场景优势更明显。

## 2E 局限与启发
- 动态预算策略仍依赖重要性估计准确度。
- 对 EvoRLoRA 的启发是：把预算调配进一步扩展到离散结构演化。

## 对本项目的关系
- 共性: 都关注 LoRA 的动态秩问题。
- 差异: EvoRank-LoRA 采用 ES 做显式离散结构变异（扩秩/减秩/重分配）。

## 关联笔记
- [[01-项目论文-EvoRank-LoRA]]
- [[10-概念-动态秩分配]]

## 权威来源
- OpenReview: https://openreview.net/forum?id=lq62uWRJjiY



## 权威来源
- OpenReview (ICLR 2023): https://openreview.net/forum?id=lq62uWRJjiY


## 三件套卡片
- [[cards/03-A-公式卡-AdaLoRA.md|公式卡]]
- [[cards/03-B-复现实验卡-AdaLoRA.md|复现实验卡]]
- [[cards/03-C-失败案例卡-AdaLoRA.md|失败案例卡]]
