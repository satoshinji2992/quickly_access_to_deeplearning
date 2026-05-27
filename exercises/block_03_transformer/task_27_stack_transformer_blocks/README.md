# task_27：多层叠加

## 问题

单层 Transformer 能力有限。多层叠加需要残差连接和归一化保持稳定。

## 你要做什么

- 实现 Pre-Norm Transformer block。
- 堆叠多层 block。
- 跑小语料 next-token 训练。

## 验收

- loss 能下降。
- 生成文本不再完全随机。

