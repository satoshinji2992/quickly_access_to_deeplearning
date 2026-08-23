# task_24：SwiGLU 前馈网络

Attention 负责 token 之间的信息交换，FFN 则在每个 token 内部变换特征。它对所有位置复用同一组参数，但不会沿序列轴把不同 token 混在一起。

SwiGLU 是 LLaMA、PaLM 等现代大模型的标准前馈结构。它不是把 ReLU 换成另一个激活函数，而是加了一扇"门"：让网络自己决定每个特征通过多少。

## 普通 FFN：一条通道

原始 Transformer 的前馈网络只有一条路径：

$$
\operatorname{FFN}(x)=W_2\,\operatorname{ReLU}(W_1x).
$$

先扩张再压回：

```text
(B,T,D) -> (B,T,4D) -> (B,T,D)
```

$d_{\text{model}}=4096$ 的大模型里，中间维度通常是 4 倍即 16384。ReLU 对每个中间特征只做一件事：大于零通过，小于零归零——一个只有开和关的闸门。

## SwiGLU：两条投影加一扇门

SwiGLU 把输入同时投影成两路：

$$
g=W_{gate}x,\qquad u=W_{up}x,
$$

$$
\operatorname{SwiGLU}(x)
=W_{down}\left[\operatorname{SiLU}(g)\odot u\right].
$$

其中 $g$ 是门控分支，$u$ 是内容分支，$\odot$ 是逐元素相乘，$W_{down}$ 把乘积投影回模型维度。数据流：

```text
x ──→ W_gate ──→ SiLU ──┐
                         ⊙ ──→ W_down ──→ 输出
x ──→ W_up ─────────────┘
```

直观理解这一步相乘：

$$
\underbrace{\operatorname{SiLU}(W_{gate}x)}_{\text{决定放行多少}}
\odot
\underbrace{W_{up}x}_{\text{放行的内容}}.
$$

门控分支像每个特征自己的**音量旋钮**，连续地控制另一路特征通过多少；ReLU 则只有"保留或归零"两个状态。名称也因此而来：**Swi** = Swish（即 SiLU），**GLU** = Gated Linear Unit（门控线性单元）——用 Swish 做门的 GLU。

![SwiGLU 的 gate、up 和 down 三次投影](assets/swiglu.png)

<div class="widget-mount" data-widget="swiglu-gate" data-title="拖一拖，看门开合"></div>

## SiLU：平滑的门

$$
\operatorname{SiLU}(z)=z\,\sigma(z),\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

与 ReLU 相比，SiLU 处处平滑，且允许小幅负值通过（$z=-1$ 时约为 $-0.27$，不是 0）。门因此可以从"几乎全关"连续拧到"全开"，中间任何开度都可用。

## 参数量：为什么中间维度取 8/3·d 而不是 4d

普通 FFN 两个矩阵，参数量约为：

$$
4d^2+4d^2=8d^2.
$$

SwiGLU 三个矩阵（$d\rightarrow m$ 两次、$m\rightarrow d$ 一次）：

$$
3dm.
$$

若仍取 $m=4d$，参数量会变成 $12d^2$，比普通 FFN 多一半。为了让两者接近，令：

$$
3dm\approx 8d^2
\qquad\Longrightarrow\qquad
m\approx \frac{8}{3}d\approx 2.67d.
$$

所以使用 SwiGLU 后中间维度不再取 $4d$。LLaMA 的 $d=4096$：

$$
4096\times\frac{8}{3}\approx 10923,
$$

再为硬件计算效率对齐（通常是 256 的倍数），LLaMA 取 `11008`。

本仓库的取值更小：`ffn.py` 的独立示例用 `hidden_dim = 2×dim`；Task 27 的模型主干用 `hidden_dim = 1.5×dim`。教学模型在 CPU 上运行，参数量优先保证跑得动；比例本身不是 SwiGLU 的结构约束。

## `ffn.py` 中的两个版本

```text
FeedForward  Linear -> GELU -> Linear
SwiGLU      w3(SiLU(w1(x)) * w2(x))
```

对应关系：

1. `w1`、`w2` 是两个不同的 `nn.Linear`，都把 `D` 投影到 `H`；
2. 只有 `w1` 分支经过 SiLU；
3. `w3` 把乘积投影回 `D`；
4. batch 和 sequence 维始终保留。

LLaMA 系模型给同样的三个矩阵起名为 `gate_proj` / `up_proj` / `down_proj`。Task 27 的 `minimind_core.py` 就采用这一套命名，并把三个投影设为 `bias=False`；`ffn.py` 使用 `nn.Linear` 的默认 bias。命名与 bias 都不改变分支结构。

这也说明为什么下面的结构不等价：

```text
Linear -> SiLU -> Linear
```

它只有一次上投影，没有独立的 up 分支，是普通两层 FFN，不是 SwiGLU。

## 为什么门控更好

ReLU FFN 对每个特征做二值决定；SwiGLU 根据当前输入连续地控制通过量，特征选择更灵活，SiLU 的平滑性也让优化更稳定。在相近参数量和计算量下，门控 FFN 普遍取得更低的语言建模损失——这是 LLaMA、PaLM 等模型采用它的直接原因。

## FFN 不会破坏 causal 性质

可把计算理解为：

```text
x[:,0,:] -> 同一个 ffn -> out[:,0,:]
x[:,1,:] -> 同一个 ffn -> out[:,1,:]
...
```

FFN 不沿 `T` 维求和，所以不会让早期位置读到未来。序列方向的信息边界仍由 attention 的 causal mask 决定。

## 运行与核对

```bash
python exercises/block_03_transformer/task_24_swiglu_ffn/ffn.py
```

输出形如：

```text
SwiGLU output: (2, 6, 32)
```

SwiGLU 的结构性质也收录在 Block 3 测试中：

```bash
python -m unittest discover -s tests -p 'test_block3.py' -v
```

输出与测试中可以看到：输入输出均为 `(B,T,D)`，两次上投影使用不同 Parameter，逐元素乘法前 shape 一致，前向结果有限且可正常反向传播。

参考：[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)。
