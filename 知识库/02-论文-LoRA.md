---
tags: [论文, LoRA, PEFT]
status: active
updated: 2026-04-05
---

# LoRA: Low-Rank Adaptation of Large Language Models

## 原文
- [Hu 等 - 2021 - LoRA Low-Rank Adaptation of Large Language Models.pdf](../论文/Hu%20等%20-%202021%20-%20LoRA%20Low-Rank%20Adaptation%20of%20Large%20Language%20Models.pdf)

## 核心观点
- 冻结预训练权重，仅训练低秩矩阵增量，实现参数高效微调。
- 通过低秩分解近似全参数更新方向。

## 2A 问题定义与动机
- 全参数微调对显存、存储和多任务部署成本过高。
- 需要一种低开销、可插拔、几乎不增加推理延迟的替代方案。

## 方法抽象
- `W' = W + BA`
- 其中 `B ∈ R^{d×r}`, `A ∈ R^{r×k}`，`r` 通常固定。

## 2B 方法与创新
- 在冻结 backbone 的前提下，只训练低秩路径参数。
- 对 Transformer 中关键线性层注入低秩适配器，减少可训练参数量。

## 2C 技术细节
- rank `r` 是核心容量超参。
- 初始化与缩放策略会影响训练稳定性与收敛速度。
- 在实际工程中常结合量化、混合精度和高效优化器使用。

## 2D 实验观察（论文层面）
- 在多项下游任务上，LoRA 以显著更低训练开销达到接近甚至可比全参微调性能。
- 适配权重可独立保存，便于多任务部署和模型复用。

## 2E 局限与启发
- 固定秩在层间与阶段上不灵活，容易出现容量错配。
- 直接启发了后续动态秩/预算分配方法（如 AdaLoRA、EvoRLoRA）。

## 对本项目的启发
- 提供了可插拔的 PEFT 骨架。
- 固定秩限制了层间与训练阶段的适配弹性，这正是 EvoRank-LoRA 要解决的问题。

## 关联笔记
- [[01-项目论文-EvoRank-LoRA]]
- [[10-概念-动态秩分配]]



## 权威来源
- arXiv: https://arxiv.org/abs/2106.09685
- Microsoft LoRA repo: https://github.com/microsoft/LoRA


## 三件套卡片
- [[cards/02-A-公式卡-LoRA.md|公式卡]]
- [[cards/02-B-复现实验卡-LoRA.md|复现实验卡]]
- [[cards/02-C-失败案例卡-LoRA.md|失败案例卡]]
