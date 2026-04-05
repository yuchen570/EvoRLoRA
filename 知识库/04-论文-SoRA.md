---
tags: [论文, SoRA, 稀疏低秩, PEFT]
status: active
updated: 2026-04-05
---

# SoRA: Sparse Low-rank Adaptation

## 原文
- [SoRA.pdf](../论文/SoRA.pdf)

## 核心观点
- 将稀疏性与低秩结合，提升参数利用效率。
- 关注参数结构化约束下的性能-效率折中。

## 2A 问题定义与动机
- 仅靠低秩并不总能充分捕获任务相关更新结构。
- 通过稀疏化与低秩结合，进一步压缩冗余参数。

## 2B 方法与创新
- 在低秩适配路径中引入稀疏结构约束。
- 同时利用“低秩全局结构”和“稀疏局部选择”。

## 2C 技术细节
- 稀疏度和秩两个维度共同影响容量与训练稳定性。
- 需要平衡结构约束强度与可优化性。

## 2D 实验观察
- 相比纯 LoRA，在一些任务可提供更优性能/效率平衡。
- 更适合对参数/显存预算敏感的场景。

## 2E 局限与启发
- 稀疏策略引入额外超参敏感性。
- 与 EvoRLoRA 组合方向：动态秩 + 稀疏约束联合搜索。

## 对本项目的关系
- SoRA 强调稀疏低秩结构。
- EvoRank-LoRA 强调训练过程中秩结构的动态进化。

## 关联笔记
- [[01-项目论文-EvoRank-LoRA]]
- [[10-概念-动态秩分配]]

## 权威来源
- ACL Anthology: https://aclanthology.org/2023.emnlp-main.252/



## 权威来源
- ACL Anthology (EMNLP 2023): https://aclanthology.org/2023.emnlp-main.252/


## 三件套卡片
- [[cards/04-A-公式卡-SoRA.md|公式卡]]
- [[cards/04-B-复现实验卡-SoRA.md|复现实验卡]]
- [[cards/04-C-失败案例卡-SoRA.md|失败案例卡]]
