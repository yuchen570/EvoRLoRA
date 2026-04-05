---
tags: [基础知识, PEFT, LoRA]
status: active
updated: 2026-04-05
---

# 基础知识：PEFT 与 LoRA 生态

## 直觉
- PEFT 的目标不是“训练最少参数”，而是“在可接受性能下最小化训练与部署成本”。
- LoRA 通过低秩增量 `BA` 在不改动主干参数的前提下逼近有效更新方向。

## 正式化
- 对于权重矩阵 `W`，LoRA 写作 `W' = W + BA`，其中 `rank(B A) <= r`。
- rank `r` 控制表达能力与开销；过小可能欠拟合，过大可能浪费预算。

## 与本项目关联
- EvoRLoRA 的核心就是把固定 `r` 拓展为“可演化的结构变量”。

## 推荐阅读
- LoRA 原文（arXiv）: https://arxiv.org/abs/2106.09685
- PEFT 文档（Hugging Face）: https://huggingface.co/docs/peft/index
- AdaLoRA（OpenReview）: https://openreview.net/forum?id=lq62uWRJjiY

## 关联笔记
- [[01-项目论文-EvoRank-LoRA]]
- [[03-论文-AdaLoRA]]
- [[40-论文对比矩阵-EvoRLoRA]]

