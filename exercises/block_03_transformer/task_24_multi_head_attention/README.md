# task_24：多头注意力

## 问题

一个 attention 头表达能力有限。多头注意力让模型从多个子空间看上下文。

## 你要做什么

- 实现 MHA。
- 实现 causal mask。
- 选做：实现 GQA。

## 验收

- 输出 shape 为 `(batch, seq_len, dim)`。
- mask 后未来 token 权重为 0。
- 支持 RoPE 接入。

