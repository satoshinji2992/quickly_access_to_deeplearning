# task_22：RoPE 位置编码

正弦位置编码把位置向量加到 embedding。RoPE（Rotary Position Embedding）换了一个入口：embedding 保持不动，attention 中的 Q、K 按位置旋转，V 不旋转。

这样做的着眼点是匹配分数。位置关系需要改变“当前 query 与哪个 key 匹配”，而 value 仍负责提供被汇总的内容。

![RoPE：不同维度对使用不同旋转频率](assets/rope.png)

---

## 把相邻两维看成一个平面

取 Q 或 K 的一对分量 $(x_0,x_1)$。位置 $m$ 对应一次二维旋转：

$$
\begin{bmatrix}x'_0\\x'_1\end{bmatrix}
=
\begin{bmatrix}
\cos(m\theta)&-\sin(m\theta)\\
\sin(m\theta)& \cos(m\theta)
\end{bmatrix}
\begin{bmatrix}x_0\\x_1\end{bmatrix}.
$$

展开第一、二行：

$$
x'_0=x_0\cos(m\theta)-x_1\sin(m\theta),
$$

$$
x'_1=x_1\cos(m\theta)+x_0\sin(m\theta).
$$

这正是代码中 `rotate_half(x)` 先得到 `(-x1,x0)`，再计算 `x*cos + rotate_half(x)*sin` 的原因。

输入通常是：

```text
x: (B,H,T,Dh)
```

旋转只发生在最后一维，输出 shape 仍为 `(B,H,T,Dh)`。

## 每个二维平面的角速度不同

维度按 `(0,1), (2,3), ...` 配对。第 `i` 对使用：

$$
\theta_i=base^{-2i/D_h},\qquad base=10000.
$$

于是位置相同，不同维度对转过的角度仍不同。`build_rope_cache` 只为每一对保存一个 cos/sin 值：

```text
cos, sin: (T,Dh/2)
```

`apply_rope` 再把每列重复两次，与成对分量对齐。图中的三个圆分别表示三个维度对，不是三个 token；箭头长短只作示意，重要的是角速度 $\omega_0,\omega_1,\omega_2$ 不同。

## 点积为何出现相对位移

把位置 $m$ 的 query 写成 $R_mq$，位置 $n$ 的 key 写成 $R_nk$：

$$
(R_mq)^\top(R_nk)
=q^\top R_m^\top R_nk
=q^\top R_{n-m}k.
$$

位置项通过 $n-m$ 进入点积。因此较准确的表述是：**RoPE 让旋转后的 Q/K 点积显式依赖相对位移**。

这不等于“模型输出对绝对位置完全不变”。内容、causal mask、上下文窗口和后续网络仍会影响结果；上式只说明旋转后的匹配分数具有怎样的位置结构。

## `start_pos` 是为缓存解码准备的

完整 forward 的位置从 0 开始。若 KV Cache 已保存 5 个 token，新 token 应使用位置 5，而不是重新使用位置 0：

```python
cos, sin = build_rope_cache(
    seq_len=1,
    head_dim=head_dim,
    start_pos=5,
)
```

两种取法的结果一致：

```text
cache(start_pos=5, seq_len=1) == cache(start_pos=0, seq_len=6)[5:6]
```

漏掉这个偏移时，代码可能仍能运行，shape 也完全正常，但 cached logits 会偏离完整 forward。Task 30 的数值等价测试会覆盖这类情形。

## 代码结构

`rope.py` 有三个函数：

```text
build_rope_cache  构造各位置、各维度对的 cos/sin
rotate_half       (x0,x1) -> (-x1,x0)
apply_rope        把旋转应用到 (B,H,T,Dh)
```

实现中的边界如下：

- `Dh` 为正偶数；
- 位置 0 的 cos 全为 1、sin 全为 0，因此向量不变；
- 位置 1 的不同列不相同；
- `apply_rope` 保持 shape 和 dtype；
- cos/sin 的 shape 与输入序列长度不符时立即报错。

运行：

```bash
python exercises/block_03_transformer/task_22_rope_position/rope.py
```

输出形如：

```text
input/output: (2, 4, 6, 8) (2, 4, 6, 8)
pair frequencies differ: True
```

RoPE 的边界也收录在 Block 3 测试中：

```bash
python -m unittest discover -s tests -p 'test_block3.py' -v
```

运行与测试会呈现这些性质：位置 0 不旋转，维度对频率不同，shape/dtype 不变，奇数 `head_dim` 会触发入参错误，`start_pos` 切片与完整 cache 对齐。

参考：[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)。
