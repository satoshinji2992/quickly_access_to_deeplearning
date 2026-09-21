# 正弦位置编码

[Attention 与 decoder-only](../task_20_transformer_theory/README.md) 中的 Q/K 先根据向量内容匹配。若既没有位置信号，也没有 causal mask 这样的非对称约束，调换 token 的顺序只会同样调换输出，模型无法分辨它们原来的位置。Causal mask 规定了“左边可见、右边不可见”，位置编码则进一步把位置与距离写进向量。原始 Transformer 的做法是为每个整数位置生成一行固定的 sin/cos 数值，再与 token embedding 相加。

## 一对维度共用一个频率

向量的相邻两维配成一对，同一对共用频率，一维取 sin，另一维取 cos。位置 `pos` 与第 `i` 对特征的数值为：

$$
PE(pos,2i)=\sin\left(pos/10000^{2i/D}\right),
$$

$$
PE(pos,2i+1)=\cos\left(pos/10000^{2i/D}\right).
$$

`D` 是 embedding 维度。维度 `(0,1)` 共用一个频率，`(2,3)` 共用另一个频率；前面的维度对变化快，越靠后变化越慢，所以一条正弦曲线无法代表整张位置表。代码把分母改写成指数形式，一次生成每对维度的频率：

```python
div_term = exp(arange(0, D, 2) * (-log(10000) / D))
angles = position * div_term
```

`div_term[i]` 就是第 `i` 对维度的角速度，`position` 与它做外积后，就得到“位置 × 维度对”的角度表。

![不同维度具有不同频率的正弦位置编码](assets/sinusoidal_position.png)

<div class="widget-mount" data-widget="pos-encoding"></div>

## 位置 0 是一个有用的对照

`pos=0` 是很方便的对照，因为此时所有角度都是 0，第 0 行必然是：

```text
PE[0] = [0,1,0,1,0,1,...]
```

若第 0 行不是这个结果，常见原因是 sin/cos 列写反，或位置编号误从 1 开始。正弦位置编码还有一个很重要的性质：对同一对维度，和角公式给出

$$
\begin{aligned}
\sin(a+b)&=\sin a\cos b+\cos a\sin b,\\
\cos(a+b)&=\cos a\cos b-\sin a\sin b.
\end{aligned}
$$

这表明位置 `pos+k` 的一对数值，可以由位置 `pos` 的数值经过一次只与 `k` 有关的线性变换得到，这也是原论文选用 sin/cos 的动机之一。

## 位置表怎样加到 embedding

`sinusoidal_position_encoding(max_len, D)` 返回 `(max_len,D)` 的完整表。当前 batch 的 token embedding 是 `(B,T,D)`，只取位置表的前 `T` 行，再借助 batch 维的广播做逐元素相加：

```python
x = token_embedding + position_table[:T][None, :, :]
# (B,T,D)       + (1,T,D) -> (B,T,D)
```

![Token embedding 与位置编码逐元素相加](assets/embedding_plus_position.png)

广播后，同一位置在所有 batch 样本中使用同一行位置编码。这张位置表保持固定，token embedding 仍然会在训练中更新。

## 当前实现要求偶数维

函数分别向 `0::2` 和 `1::2` 写入 sin/cos。每个 sin 列都需要一个相邻的 cos 列，所以这份实现的输入约束是：

```text
max_len > 0
D > 0
D % 2 == 0
```

这是 `position.py` 的接口约束，并不意味着所有位置编码都必须使用偶数维。直接运行脚本，可以用第 0 行核对实现：

```bash
python exercises/block_03_transformer/task_21_sinusoidal_position/position.py
```

输出中的第 0 行应与手算结果一致：

```text
shape: (4, 8)
position 0: [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
```

MiniMind 没有把这张表加到 embedding，而是采用 [RoPE](../task_22_rope_position/README.md)：多频率 sin/cos 仍然保留，它们的作用对象改成了 Attention 的 Q/K。

参考：[Attention Is All You Need，第 3.5 节](https://arxiv.org/abs/1706.03762)。
