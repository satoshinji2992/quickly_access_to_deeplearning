# task_21：交叉注意力实现

## 问题

交叉注意力中，query 来自当前序列，key/value 来自另一段 memory。

## 你要做什么

- 实现 `CrossAttention`。
- 跑一个 toy retrieval 例子。

## 验收

- 输出 shape 为 `(batch, query_len, dim)`。
- attention 权重 shape 为 `(batch, heads, query_len, memory_len)`。

