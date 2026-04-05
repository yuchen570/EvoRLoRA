---
tags: [论文, ES, 大规模优化]
status: active
updated: 2026-04-05
---

# Evolution Strategies at the Hyperscale

## 原文
- [Evolution Strategies at the Hyperscale.pdf](../论文/Evolution%20Strategies%20at%20the%20Hyperscale.pdf)

## 核心观点
- 探索 ES 在超大规模设置下的可扩展优化能力。
- 强调群体规模扩展和无反向传播优化的实践可行性。

## 2A 问题定义与动机
- 朴素 ES 在 GPU 大规模场景下存在算术强度不足导致的效率瓶颈。
- 需要结构化扰动以提升吞吐并保持优化质量。

## 2B 方法与创新
- 提出 EGGROLL 等结构化低秩扰动思路，提高大规模训练效率。
- 同时给出高维理论分析，讨论与经典 ES 的一致性条件。

## 2C 技术细节
- 重点在扰动结构化设计、批量并行效率和高维收敛性质。
- 对“参数维度极高 + 大群体规模”条件下的可行性有针对性优化。

## 2D 实验观察
- 在大模型与大群体规模下实现显著吞吐提升。
- 说明 ES 类方法在现代硬件与大模型场景仍有增长空间。

## 2E 局限与启发
- 方法复杂度与实现门槛高于经典 ES。
- 对 EvoRLoRA 的启发：若外层结构搜索规模扩大，可引入结构化扰动加速。

## 对本项目的关系
- 为 EvoRank-LoRA 的外层 ES 设计提供规模化思路。
- 可用于指导候选结构采样规模、并行评估和稳定性策略。

## 关联笔记
- [[01-项目论文-EvoRank-LoRA]]
- [[11-概念-进化策略]]

## 权威来源
- arXiv: https://arxiv.org/abs/2511.16652



## 权威来源
- arXiv: https://arxiv.org/abs/2511.16652


## 三件套卡片
- [[cards/07-A-公式卡-ES-Hyperscale.md|公式卡]]
- [[cards/07-B-复现实验卡-ES-Hyperscale.md|复现实验卡]]
- [[cards/07-C-失败案例卡-ES-Hyperscale.md|失败案例卡]]
