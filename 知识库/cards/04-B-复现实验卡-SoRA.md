---
tags: [论文卡片, 复现实验卡, SoRA]
status: active
updated: 2026-04-05
---

# 复现实验卡：
SoRA

## 关联论文
- 
[[04-论文-SoRA]]
- 图谱入口：[[图谱-EvoRLoRA论文网络]]

## 复现目标
- 复现主表核心指标趋势与参数效率收益。

## 实验配置模板
- 模型：`TODO(backbone)`
- 数据集：`TODO(dataset)`
- 训练步数：`TODO(steps)`
- 优化器/LR：`TODO(optim/lr)`
- 结构/预算超参：`TODO(rank/budget)`

## 执行清单
- [ ] 至少 3 个随机种子
- [ ] 对齐 baseline 总步数
- [ ] 保存日志与 checkpoint
- [ ] 回填 [[40-论文对比矩阵-EvoRLoRA]]

## 输出要求
- 主指标均值±方差
- 显存峰值/时长/参数量
- 相对 baseline 的收益与退化说明

## 补充资源
| 类型 | 标题 | 链接 | 日期 | 关联说明 |
|---|---|---|---|---|
| 论文 | 主要来源 | 
https://aclanthology.org/2023.emnlp-main.252/
 | 2026-04-05 | 复现实验设置依据。 |
| 资源 | 对照来源 | 
https://arxiv.org/abs/2106.09685
 | 2026-04-05 | 验证口径一致性。 |
| 文档 | 工程参考 | 
https://huggingface.co/docs/peft/index
 | 2026-04-05 | 实现细节参考。 |