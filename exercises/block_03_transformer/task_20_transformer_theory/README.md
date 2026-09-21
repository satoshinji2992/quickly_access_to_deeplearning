# Attention 与 decoder-only

[入口与出口](../task_25_embedding_lm_head/README.md) 已经把 token ids 查表变成 `X: (B,T,D)`。如果同一个 MLP 独立处理每个位置，末尾 token 永远读不到前文。Self-attention 在任意两个位置之间计算匹配分，于是一个位置可以直接取回另一个位置的信息。代价是产生 `T×T` 的关系矩阵。RNN 也能传递前文，但较远的信息需要经过多次递推，序列方向的计算也难以完全并行。

![Scaled dot-product self-attention 的数据流和 shape](assets/self_attention.png)

## Q、K、V 各做什么

三组参数各不相同的线性投影，把 $X$ 变成三种用途不同的向量：

```text
Q = X @ Wq    # 当前位置用什么去匹配
K = X @ Wk    # 各位置用什么接受匹配
V = X @ Wv    # 匹配后取回的内容
```

一个 attention head 把这三者组合为

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_{head}}}+M\right)V.
$$

`QKᵀ` 的第 `t` 行、第 `j` 列是 query `t` 和 key `j` 的匹配分。Softmax 沿 key 方向把每行变成权重，再按权重汇总 V：Q/K 决定从哪里取，V 决定取回什么。点积由 `D_head` 项乘积相加。若 Q、K 各维尺度相近，分数的方差会随 `D_head` 增大；除以 $\sqrt{d_{head}}$ 可以避免 Softmax 过早变得尖锐。

公式中的 $M$ 是加法 mask。它不必与 score 完全同 shape，只要能广播到 `(B,n_heads,T_q,T_k)`：允许的位置加 0，禁止的位置加负无穷。具体的 causal mask、padding mask 与边界处理留到完整 Attention 实现时再看。

<div class="widget-mount" data-widget="qkv-flow"></div>

## 多个 head 使用多组投影

多个 head 使用不同的投影参数，并非重复同一次计算。以 `D=32, n_heads=4` 为例，`D_head=8`，标准 MHA 的 shape 变化是：

```text
Q, K, V:     (B,T,32) -> (B,4,T,8)
scores:                   (B,4,T,T)
head outputs:             (B,4,T,8)
concat:                   (B,T,32)
```

拼接后的宽度回到 32，而不同投影让各 head 有机会形成不同的匹配方式。某个 head 有时会呈现可解释的模式，但模型并没有预先把“语法”或“指代”分配给固定 head。

## Causal mask 把 Self-attention 变成前缀读取

Next-token 训练会同时把整段序列放进张量，因此位置 2 在数据上确实与位置 3 并存，只是 Attention 不允许它读取这个未来位置。`T=4` 时的可见关系是：

```text
q0 -> k0
q1 -> k0 k1
q2 -> k0 k1 k2
q3 -> k0 k1 k2 k3
```

这个可见范围正好对应含对角线的下三角 mask。固定前缀后任意改动未来 token，前缀位置的输出应当保持不变；比起只看一张三角形图，这项数值性质更容易暴露 mask 方向写反的错误。

## Encoder–decoder 和 decoder-only

《Attention Is All You Need》的主要场景是机器翻译：Encoder 读取源语言，decoder 一边用 masked self-attention 处理已有的目标语言，一边通过 cross-attention 读取 encoder 输出。

```text
source -> Encoder ------------------+
                                      -> Decoder -> target logits
shifted target -> masked self-attn --+   (cross-attention 读 encoder)
```

本仓库的小型语言模型只根据已有 token 继续文本，因而使用 decoder-only 结构：

```text
已有 token -> causal self-attention -> next-token logits
```

![Encoder-decoder 与 decoder-only](assets/encoder_decoder_vs_decoder_only.png)

Decoder-only 没有 encoder 和 cross-attention，Q/K/V 都来自当前 token 序列。图左的 decoder 中，一种 Attention 读取已生成前缀，另一种读取 encoder context；图右只保留前者。Causal mask 能区分前后方向，却不会直接给出绝对位置或 token 间距。[正弦位置编码](../task_21_sinusoidal_position/README.md) 先从一种直接加到 embedding 上的办法说起。

参考：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[PyTorch SDPA API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。
