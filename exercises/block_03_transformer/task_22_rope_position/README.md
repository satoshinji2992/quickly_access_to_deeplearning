# RoPE：把位置写进 Q/K

[正弦位置编码](../task_21_sinusoidal_position/README.md) 把一行 sin/cos 数值加到 token embedding 上，RoPE（Rotary Position Embedding）则把同类的多频率 sin/cos 放进 Attention 内部。它不改 embedding，而是按位置旋转 Q 和 K，V 保持原样。这样的分工与三者的作用相合：Q/K 的点积决定匹配分数，V 负责提供最后汇总的内容。

## 相邻两维组成一个平面

取 Q 或 K 的一对相邻分量 $(x_0,x_1)$，把它看成平面上的一个点。位置 $m$ 会让这个点旋转 $m\theta$：

$$
\begin{bmatrix}x'_0\\x'_1\end{bmatrix} =
\begin{bmatrix}
\cos(m\theta)&-\sin(m\theta)\\
\sin(m\theta)& \cos(m\theta)
\end{bmatrix}
\begin{bmatrix}x_0\\x_1\end{bmatrix}.
$$

把矩阵乘法展开，两个新分量分别是

$$
x'_0=x_0\cos(m\theta)-x_1\sin(m\theta),\qquad
x'_1=x_1\cos(m\theta)+x_0\sin(m\theta).
$$

`rotate_half(x)` 先把 `(x0,x1)` 变成 `(-x1,x0)`，正好对应上式中与 sin 相乘的部分，所以代码可以紧凑地写成：

```python
rotated = x * cos + rotate_half(x) * sin
```

输入常用 `(B,n_heads,T,D_head)`，旋转只在最后一维中成对进行，不会改变张量 shape。

![RoPE：不同维度对使用不同旋转频率](assets/rope.png)

<div class="widget-mount" data-widget="rope-rotate"></div>

## 每对维度的角速度不同

维度按 `(0,1), (2,3), ...` 配对，每一对都有自己的角速度。第 `i` 对使用

$$
\theta_i=base^{-2i/D_h},\qquad base=10000.
$$

位置同样是 $m$，不同维度对也会转过不同角度。`build_rope_cache` 没有为两个相邻分量重复存值，而是每对维度只存一个 cos/sin：

```text
cos, sin: (T,Dh/2)
```

`apply_rope` 使用时再将每列重复两次，与成对分量对齐。图中的三个圆因而代表三对特征，不是三个 token。

例如 `D_head=4` 时只有两对分量，角速度分别是 $10000^0=1$ 和 $10000^{-1/2}=0.01$。同在位置 2，第一对转过 2 弧度，第二对只转过 0.02 弧度。如果两对原本都是 `[1,0]`，旋转后的四维向量约为 `[-0.4161,0.9093,0.9998,0.0200]`。位置没有变，差别完全来自各特征对的频率；把这两对都旋转 2 弧度，就不再是这里定义的 RoPE。

## 点积里出现了相对位移

为了看清位置如何进入 Attention 分数，把位置 $m$ 的 query 写成 $R_mq$，位置 $n$ 的 key 写成 $R_nk$，两者的点积可以化为

$$
(R_mq)^\top(R_nk)
=q^\top R_m^\top R_nk
=q^\top R_{n-m}k.
$$

旋转对点积的影响以 $n-m$ 的形式出现，这就是“点积依赖相对位移”的准确含义。它并不表示分数只由距离决定：$q$ 和 $k$ 的内容仍在式子里，causal mask、上下文窗口和后续网络也会影响最终输出。

## 缓存解码时，位置要接着往后数

完整 forward 从位置 0 开始；缓存中已有 `past_len` 个位置时，新 token 应使用位置 `past_len`。若每步都从 0 起算，shape 和 cache 长度仍然正常，旋转角却会悄悄错开。`start_pos` 如何随 cache 更新，放在 [KV Cache](../task_30_kv_cache/README.md) 中结合完整解码流程再看。

## 代码对应

`rope.py` 的三个函数分别负责构造角度表、产生 `(-x1,x0)`，以及把旋转应用到 `(B,n_heads,T,D_head)`。特征必须两两配对，所以当前实现要求 `D_head` 为正偶数，cos/sin 的序列长度也要与输入一致。运行脚本后会看到 shape 保持不变，各特征对的频率也并不相同：

```bash
python exercises/block_03_transformer/task_22_rope_position/rope.py
```

```text
input/output: (2, 4, 6, 8) (2, 4, 6, 8)
pair frequencies differ: True
```

RoPE 解决的只是 Q/K 怎样携带位置，[完整的 causal attention](../task_23_causal_attention/README.md) 还需要 Q/K/V 投影、mask、多头拼接和输出投影。

参考：[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)。
