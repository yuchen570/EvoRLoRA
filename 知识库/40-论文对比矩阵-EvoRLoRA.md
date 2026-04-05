---
tags: [对比矩阵, LoRA, EvoRLoRA, 文献综述]
status: active
updated: 2026-04-05
---

# 论文对比矩阵（LoRA 系列 + ES）

| 论文 | 核心问题 | 主要方法 | 结构是否动态 | 备注 |
|---|---|---|---|---|
| LoRA | 降低全参微调成本 | 固定低秩增量 `BA` | 否 | PEFT 基石方法 |
| AdaLoRA | 固定秩预算分配不优 | 自适应预算分配 | 是（预算层面） | 动态秩方向代表基线 |
| SoRA | 提升参数利用效率 | 稀疏 + 低秩联合 | 部分 | 偏结构稀疏化 |
| LoRA-GA | LoRA 收敛慢问题 | 梯度近似增强 | 否（主打训练策略） | 可与动态结构方法互补 |
| EvoRank-LoRA | 固定秩层间/时序失配 | ES 外环 + 梯度内环 | 是（显式离散结构） | 支持扩秩/减秩/重分配 |
| ES 2017 | RL 可扩展优化替代 | 黑盒 ES 并行优化 | - | 提供 ES 方法论基础 |
| ES Hyperscale | ES 在大模型大群体下效率瓶颈 | 低秩扰动结构化 ES | - | 提供规模化启发 |

## 参考链接
- LoRA: https://arxiv.org/abs/2106.09685
- AdaLoRA: https://openreview.net/forum?id=lq62uWRJjiY
- SoRA: https://aclanthology.org/2023.emnlp-main.252/
- LoRA-GA: https://arxiv.org/abs/2407.05000
- ES 2017: https://arxiv.org/abs/1703.03864
- ES Hyperscale: https://arxiv.org/abs/2511.16652

## 关联笔记
- [[21-paper-learning-学习总笔记]]
- [[00-总览-EvoRLoRA]]

