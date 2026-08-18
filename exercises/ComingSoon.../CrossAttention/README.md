# Cross-Attention

Cross-attention 让当前序列从另一段序列中取回信息。与 self-attention 的区别只在 Q、K、V 的来源：

```text
self-attention:  Q, K, V 都由同一序列投影得到
cross-attention: Q 来自当前序列，K, V 来自 memory
```

设 `query` 为 `(B, T_q, D)`，`memory` 为 `(B, T_m, D)`，分头后的权重矩阵是 `(B, H, T_q, T_m)`，输出仍为 `(B, T_q, D)`。`T_q` 与 `T_m` 不必相等。

在 encoder–decoder 翻译模型中，decoder 的隐状态提供 Q，encoder 输出提供 K/V。多模态模型和带条件的扩散模型也会使用同样的结构。

本页只记录定义和 shape，尚未提供与主线任务同等的代码和测试。公式可参考 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。
