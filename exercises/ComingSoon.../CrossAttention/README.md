# Cross-Attention

Cross-attention 的核心是让一段序列去读取另一段序列.

Self-attention 里, Q、K、V 都来自同一段序列.

Cross-attention 里:

```text
Q 来自当前序列
K, V 来自另一段 memory
```

这会出现在很多地方:

- encoder-decoder 翻译模型.
- 图文多模态模型.
- 扩散模型里的条件控制.

核心区别:

```text
self-attention:
Q, K, V 都来自同一段序列

cross-attention:
Q 来自当前序列
K, V 来自另一段 memory
```

举个翻译模型的例子:

```text
encoder 读英文句子 -> 得到 memory
decoder 生成中文句子 -> 用 cross-attention 读取 memory
```

所以 cross-attention 更像“查资料”. 当前序列提出 query, 另一段序列提供 key/value.

代码里的 `cross_attention.py` 是一个最小实现, 可以用来检查 shape:

```text
query  : (batch, query_len, dim)
memory : (batch, memory_len, dim)
output : (batch, query_len, dim)
attn   : (batch, heads, query_len, memory_len)
```
