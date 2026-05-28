# MQA / GQA

普通 Multi-Head Attention 里, Q/K/V 都有同样数量的 heads.

比如:

```text
n_heads = 32
```

那么 Q、K、V 都有 32 个 head.

这在训练时没什么问题, 但推理时会带来一个麻烦: KV Cache 很大.

## MQA

MQA(Multi-Query Attention)让所有 Q heads 共享同一组 K/V.

```text
n_heads = 32
n_kv_heads = 1
```

KV Cache 会小很多, 但表达能力可能受影响.

## GQA

GQA(Grouped-Query Attention)折中一点.

多个 Q heads 共享一组 K/V:

```text
n_heads = 32
n_kv_heads = 8
```

每 4 个 Q heads 共用 1 个 KV head.

很多 LLaMA 风格模型会用 GQA, 因为它在推理速度和效果之间比较均衡.

## 和 KV Cache 的关系

KV Cache 的大小大致和 `n_kv_heads` 成正比.

所以减少 KV heads, 推理时显存和带宽压力都会下降.

这就是大模型喜欢 MQA/GQA 的主要原因.
