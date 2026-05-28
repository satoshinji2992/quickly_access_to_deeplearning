# KV Cache

自回归生成是一 token 一 token 往外吐.

如果每生成一个新 token, 都把整段上下文重新算一遍 attention, 会浪费很多计算.

KV Cache 的想法很简单: 过去 token 的 K 和 V 不变, 可以存起来.

## 为什么只缓存 K/V?

生成第 $t$ 个 token 时, 当前 token 的 Q 是新的.

但前面 token 的 K/V 已经算过:

```text
past K: 以前所有 token 的 key
past V: 以前所有 token 的 value
new Q : 当前 token 的 query
```

新的 Q 去看历史 K/V, 就能得到当前步的 attention 输出.

这样每一步不用重新计算全部历史 token 的 K/V.

## cache 的形状

常见形状:

```text
k_cache: (batch, n_kv_heads, past_len, head_dim)
v_cache: (batch, n_kv_heads, past_len, head_dim)
```

如果是多层模型, 每一层都要有自己的 cache.

## 容易错的地方

- RoPE 的 position 不能每步都从 0 开始.
- GQA 下 `n_heads` 和 `n_kv_heads` 不一样.
- cache 太长时要考虑 `max_seq_len` 或滑动窗口.
- batch 里不同样本长度不同时, mask 会更麻烦.

KV Cache 不改变模型训练目标, 它主要影响推理速度.
