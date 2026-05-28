# task_23: Causal Attention 与 GQA

MiniMind 主线需要的是 causal self-attention.

不是 cross-attention.

这里的 Q/K/V 都来自同一段 token hidden states. 区别在于, 每个位置只能看自己和前面的 token.

## 一. Self-Attention 在算什么?

输入:

```text
x.shape = (batch, seq_len, dim)
```

线性投影得到:

```text
Q, K, V
```

然后:

$$
\mathrm{score} = \frac{QK^\top}{\sqrt{d}}
$$

除以 $\sqrt{d}$ 是为了让 score 的尺度别随着 head_dim 变大而飘得太厉害.

再加 causal mask, softmax, 乘 V.

## 二. 为什么要多头?

一个 head 只能从一个子空间看上下文.

多头就是把 `dim` 拆成几份:

```text
dim = n_heads * head_dim
```

每个 head 各自算 attention, 最后 concat 回来, 再过一个输出投影.

## 三. GQA 是什么?

GQA(Grouped Query Attention)会让 Q 的 head 数多一些, K/V 的 head 数少一些.

比如:

```text
n_heads = 8
n_kv_heads = 2
```

那每组 K/V 会服务多个 Q head.

这样推理时 KV cache 更小, 速度和显存都会舒服一点.

当前 `mha.py` 先给了普通 MHA 版本. 后面 MiniMind Core 会把 GQA 接进去.

## 四. 检查什么?

- 输出 shape 仍然是 `(batch, seq_len, dim)`.
- causal mask 后, 未来位置不能被看到.
- RoPE 应该作用在 Q/K 上, 不作用在 V 上.
