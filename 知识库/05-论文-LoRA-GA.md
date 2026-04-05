---
tags: [论文, LoRA-GA, PEFT]
status: active
updated: 2026-04-05
---

# LoRA-GA: Low-Rank Adaptation with Gradient Approximation

## 原文
- [LoRA-GA.pdf](../论文/LoRA-GA.pdf)

## 核心观点
- 通过梯度近似改善 LoRA 训练效率或收敛行为。
- 属于 LoRA 训练策略改进路线。

## 2A 问题定义与动机
- LoRA 虽节省每步训练成本，但在某些任务存在收敛较慢问题。
- 希望用更有效的梯度近似提升训练效率和最终性能。

## 2B 方法与创新
- 在低秩适配训练中引入梯度近似机制。
- 关注“训练过程效率”而非仅结构容量分配。

## 2C 技术细节
- 关键在于近似质量与计算开销的平衡。
- 可作为内环优化增强模块，与外层结构搜索框架组合。

## 2D 实验观察
- 在文中任务设置下，相比标准 LoRA 可提升收敛速度或结果质量。
- 价值在于缩短达到同等性能的训练成本。

## 2E 局限与启发
- 近似误差可能在复杂任务上带来不稳定性。
- 对 EvoRLoRA 的启发：内环可用 LoRA-GA 思想，外环保持 ES 结构演化。

## 对本项目的关系
- 可作为 EvoRank-LoRA 的内层训练增强手段候选。
- 与动态秩结构搜索形成“参数优化 + 结构优化”互补。

## 关联笔记
- [[01-项目论文-EvoRank-LoRA]]
- [[10-概念-动态秩分配]]

## 权威来源
- arXiv: https://arxiv.org/abs/2407.05000



## 权威来源
- arXiv: https://arxiv.org/abs/2407.05000


## 三件套卡片
- [[cards/05-A-公式卡-LoRA-GA.md|公式卡]]
- [[cards/05-B-复现实验卡-LoRA-GA.md|复现实验卡]]
- [[cards/05-C-失败案例卡-LoRA-GA.md|失败案例卡]]
