---
tags: [论文, ES, 强化学习, 优化]
status: active
updated: 2026-04-05
---

# Evolution Strategies as a Scalable Alternative to RL (2017)

## 原文
- [Salimans 等 - 2017 - Evolution Strategies as a Scalable Alternative to Reinforcement Learning(1).pdf](../论文/Salimans%20等%20-%202017%20-%20Evolution%20Strategies%20as%20a%20Scalable%20Alternative%20to%20Reinforcement%20Learning(1).pdf)

## 核心观点
- ES 可作为黑盒优化方法，具备高并行扩展潜力。
- 通过低通信开销策略实现大规模并行训练。

## 2A 问题定义与动机
- 策略梯度与值函数方法在长时序、并行扩展、通信效率上存在工程挑战。
- ES 以黑盒优化方式规避部分梯度路径依赖。

## 2B 方法与创新
- 通过参数扰动采样估计更新方向。
- 采用高并行 worker 架构与通信压缩思路提升扩展性。

## 2C 技术细节
- 核心是噪声采样、奖励聚合和参数更新规则。
- 在分布式条件下，通信策略直接决定可扩展上限。

## 2D 实验观察
- 在 RL 基准上展示了可行性和较强并行效率。
- 说明 ES 在某类任务上可作为梯度法替代或补充。

## 2E 局限与启发
- 样本效率和噪声鲁棒性并非在所有任务都占优。
- 对 EvoRLoRA 的价值在于“离散结构变量可由 ES 外环优化”。

## 对本项目的关系
- 提供了“离散结构变量可用 ES 优化”的方法论基础。
- EvoRank-LoRA 将 ES 用于外层结构搜索，而非替代梯度训练。

## 关联笔记
- [[01-项目论文-EvoRank-LoRA]]
- [[11-概念-进化策略]]

## 权威来源
- arXiv: https://arxiv.org/abs/1703.03864



## 权威来源
- arXiv: https://arxiv.org/abs/1703.03864


## 三件套卡片
- [[cards/06-A-公式卡-ES-2017.md|公式卡]]
- [[cards/06-B-复现实验卡-ES-2017.md|复现实验卡]]
- [[cards/06-C-失败案例卡-ES-2017.md|失败案例卡]]
