---
tags: [知识库, EvoRLoRA, 文献综述]
status: active
---

# EvoRLoRA 论文知识库总览

## 研究主线
- 目标: 在 LoRA 微调中实现动态秩分配，兼顾性能与参数效率。
- 核心方法: `EvoRank-LoRA`，用进化策略（ES）做离散结构搜索，用梯度下降做连续参数更新。
- 对比对象: `LoRA / AdaLoRA / SoRA / LoRA-GA`。

## 论文入口
- [[01-项目论文-EvoRank-LoRA]]
- [[02-论文-LoRA]]
- [[03-论文-AdaLoRA]]
- [[04-论文-SoRA]]
- [[05-论文-LoRA-GA]]
- [[06-论文-Evolution-Strategies-2017]]
- [[07-论文-Evolution-Strategies-Hyperscale]]

## 概念入口
- [[10-概念-动态秩分配]]
- [[11-概念-进化策略]]
- [[12-基础知识-PEFT与LoRA生态]]
- [[13-基础知识-低秩分解与秩动态化]]
- [[14-基础知识-进化策略ES与结构搜索]]

## 后续可补充
- 实验复现日志（按日期）
- 超参数对比表（任务/模型/初始秩/最大秩/预算）
- 失败案例与负结果



## 工作流
- [[20-阅读工作台]]
- [[21-paper-learning-学习总笔记]]

## 对比与图谱
- [[40-论文对比矩阵-EvoRLoRA]]
- [[图谱-EvoRLoRA论文网络]]

## 深读优先路径
1. [[21-paper-learning-学习总笔记]]
2. [[40-论文对比矩阵-EvoRLoRA]]
3. [[01-项目论文-EvoRank-LoRA]]
4. [[图谱-EvoRLoRA论文网络]]
