---
tags: [论文卡片, 公式卡, LoRA-GA]
status: active
updated: 2026-04-05
---

# 公式卡：
LoRA-GA

## 关联论文
- 
[[05-论文-LoRA-GA]]
- 图谱入口：[[图谱-EvoRLoRA论文网络]]

## 核心公式/记号
- 记录 3-5 个关键公式：定义式、目标函数、更新规则。
- 关键变量统一到主论文记号，避免实现歧义。

## 公式到实现映射
- 代码文件建议对应：`evo_rank_lora.py`、`rank_evolution_controller.py`、`train_integration.py`。
- 维护“公式符号 -> 配置项 -> 代码变量”三列映射。

## 校验清单
- [ ] 公式符号一致
- [ ] 超参含默认值和范围
- [ ] 公式有直觉解释

## 补充资源
| 类型 | 标题 | 链接 | 日期 | 关联说明 |
|---|---|---|---|---|
| 论文 | 主要来源 | 
https://arxiv.org/abs/2407.05000
 | 2026-04-05 | 原始公式来源。 |
| 资源 | 对照来源 | 
https://arxiv.org/abs/2106.09685
 | 2026-04-05 | 交叉验证定义。 |
| 文档 | 工程参考 | 
https://huggingface.co/docs/peft/main/en/developer_guides/lora
 | 2026-04-05 | 公式到实现映射。 |