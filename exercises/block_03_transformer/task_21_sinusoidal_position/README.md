# task_21: Sinusoidal Position Encoding

Attention 本身不知道顺序.

你把一句话里的 token 打乱, 只要不加位置信息, attention 看到的顺序差别就不明显.

原始 Transformer 用的是 sinusoidal position encoding.

它不是训练出来的参数, 而是一张按公式算出来的位置表.

## 一. 公式长什么样?

对位置 $pos$ 和维度 $i$:

$$
PE(pos, 2i)=\sin\left(pos / 10000^{2i/d_{model}}\right)
$$

$$
PE(pos, 2i+1)=\cos\left(pos / 10000^{2i/d_{model}}\right)
$$

偶数维用 sin, 奇数维用 cos.

不同维度的波长不同, 有的变化快, 有的变化慢. 这样一个位置就能被编码成一串有规律的数字.

## 二. 它怎么接进模型?

原始 Transformer 会把位置编码直接加到 token embedding 上:

```text
x = token_embedding + position_encoding
```

shape 是:

```text
position_encoding: (max_len, dim)
token_embedding : (batch, seq_len, dim)
```

取前 `seq_len` 个位置后, 广播加到 batch 上.

## 三. 为什么它是 RoPE 的前置知识?

Sinusoidal position encoding 把位置作为一个向量加进输入.

RoPE 则换了思路: 不直接加位置, 而是在 attention 里旋转 Q/K.

但 RoPE 里用到的频率、sin/cos、不同维度对应不同周期, 都和 sinusoidal 这套想法有关.

所以先写这个.

## 四. 你要写什么?

当前文件是 `position.py`.

实现:

```python
sinusoidal_position_encoding(max_len, dim)
```

检查:

- 输出 shape 是 `(max_len, dim)`.
- 偶数维是 sin.
- 奇数维是 cos.
- 第 0 个位置的 sin 维是 0, cos 维是 1.

下一关看 RoPE.
