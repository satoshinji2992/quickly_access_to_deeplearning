# SwiGLU 前馈网络

[Causal Attention](../task_23_causal_attention/README.md) 把其他位置的信息汇总到当前 token，FFN 再处理这个 token 内部的特征。它对所有位置复用同一组参数，却不沿序列轴混合 token；这也是 Attention 与 FFN 在 decoder block 中的基本分工。

## 普通 FFN 先变宽，再变回原宽度

原始 Transformer 的 FFN 由两层线性变换组成，中间接 ReLU：

$$
\operatorname{FFN}(x)=W_2\,\operatorname{ReLU}(W_1x).
$$

```text
(B,T,D) -> (B,T,D_ff) -> (B,T,D)
```

`D_ff` 通常大于 `D`，第一层把特征展开到更宽的空间，第二层再投影回模型宽度。原始 Transformer 用 ReLU，负值被截为 0，正值保留原幅度，它并不是只输出 0 或 1 的二值开关。本页 `FeedForward` 对照类改用了同样常见的 GELU，下面比较的重点是“单路上投影”和“两路相乘”，不是用实验比较 ReLU 与 GELU。

## SwiGLU 在中间加了一条调制分支

SwiGLU 不再只用一路上投影，而是同时算出 gate 和 up 两组中间特征：

$$
g=W_{gate}x,\qquad u=W_{up}x,
$$

$$
\operatorname{SwiGLU}(x)
=W_{down}\left[\operatorname{SiLU}(g)\odot u\right].
$$

```text
x ──→ W_gate ──→ SiLU ──┐
                         ⊙ ──→ W_down ──→ 输出
x ──→ W_up ────────────┘
```

`W_up x` 是被调制的内容，`SiLU(W_gate x)` 与它在每个中间特征上逐元素相乘。假如某一维的两路值分别为 3 和 0.2，乘积就是 0.6。由于调制值可以为负，也可以大于 1，“门”只是对这条乘法分支的形象称呼，并非概率开关。

![SwiGLU 的 gate、up 和 down 三次投影](assets/swiglu.png)

<div class="widget-mount" data-widget="swiglu-gate"></div>

## SiLU 不会把负半轴全部截断

$$
\operatorname{SiLU}(z)=z\,\sigma(z),\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

从公式可以看出，SiLU 是平滑函数。例如 $\operatorname{SiLU}(-1)\approx-0.27$，而 $\operatorname{ReLU}(-1)=0$；当正值增大时，SiLU 也会继续增大，并不被限制在 0 到 1 之间。SwiGLU 沿用了 GLU 的乘法结构，只是把其中的激活换成了 Swish/SiLU。

## 相同参数量下的中间宽度

两种 FFN 若取相同的中间宽度，SwiGLU 会因多一次上投影而拥有更多参数。假设普通 FFN 使用 `D_ff=4D`，忽略 bias 后参数量约为

$$4D^2+4D^2=8D^2.$$

SwiGLU 有两次 `D -> D_ff` 和一次 `D_ff -> D`，参数量约为 $3DD_{ff}$。要让两种结构的矩阵参数量接近，可以令

$$
3DD_{ff}\approx8D^2
\quad\Longrightarrow\quad
D_{ff}\approx\frac{8}{3}D.
$$

实际模型还会为硬件效率把宽度取整到合适倍数，所以 $8D/3$ 只是对齐参数预算时的参考。仓库中的 CPU 小模型直接使用较小的 `hidden_dim`。Shazeer 2020 在可比参数预算下比较了 GLU 变体，论文链接放在页底；那些实验结果不是 SwiGLU 对任意任务都更好的保证。

## 代码里的两种 FFN

`ffn.py` 同时保留了普通两层 FFN 和 SwiGLU，便于直接对照两条数据流：

```text
FeedForward  Linear -> GELU -> Linear
SwiGLU      w3(SiLU(w1(x)) * w2(x))
```

`w1` 和 `w2` 是两个不同的上投影，只有 `w1` 分支经过 SiLU，`w3` 再把两路乘积投影回 `D`。这些操作都只作用于最后一维，batch 和 sequence 维保持原样：

```text
x[:,0,:] -> 同一个 FFN -> out[:,0,:]
x[:,1,:] -> 同一个 FFN -> out[:,1,:]
```

由于不同位置之间没有数据交叉，FFN 不会让早期位置读到未来，序列方向的边界仍然由 Attention mask 决定。运行脚本可以看到 SwiGLU 保持了输入 shape：

```bash
python exercises/block_03_transformer/task_24_swiglu_ffn/ffn.py
```

```text
SwiGLU output: (2, 6, 32)
```

Attention 和 FFN 都能将 `(B,T,D)` 变换回同一 shape，因而可以放入残差支路。[组装 Decoder Block](../task_26_decoder_blocks/README.md) 会用 RMSNorm 和残差连接把两者串起来。

参考：[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)。
