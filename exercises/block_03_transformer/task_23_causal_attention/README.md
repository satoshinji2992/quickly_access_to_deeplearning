# Causal Attention 与 GQA

[RoPE](../task_22_rope_position/README.md) 只负责旋转 Q/K，一层可运行的 self-attention 还要把 Q/K/V 投影、causal mask、padding mask 和多头输出连在一起。Decoder-only 模型的 Q/K/V 都来自同一段 hidden states，但读取范围受到两层限制：位置 `t` 只能读取 `0..t`，batch 中用来补齐的 PAD 也不能成为有效 key。

## Mask 放在 Softmax 前

一个 head 会先用缩放点积计算 query 位置 $t$ 与 key 位置 $j$ 的匹配分数：

$$
s_{tj}=\frac{q_tk_j^\top}{\sqrt{d_{head}}}.
$$

将 mask 加到分数上之后才做 Softmax，并用得到的权重汇总 value：

$$
a_{tj}=\operatorname{softmax}_j(s_{tj}+M_{tj}),\qquad
o_t=\sum_j a_{tj}v_j.
$$

Mask 必须放在 Softmax 之前。如果禁止位置先参与归一化，它们已经分走了概率；事后将对应权重或输出乘为 0，无法把这部分权重还给合法位置。

![Causal mask：query i 只能读取 key j≤i](assets/causal_mask.png)

<div class="widget-mount" data-widget="causal-mask"></div>

## 张量在多个 head 之间怎样拆分

`MultiHeadSelfAttention.forward` 接收 `x: (B,T,D)`。以 `D=32`、4 个 query heads 和 2 个 KV heads 为例，每个 head 的宽度是 8，三组投影的 shape 为：

```text
head_dim = 32 / 4 = 8

Q: (B,T,32) -> (B,4,T,8)
K: (B,T,16) -> (B,2,T,8)
V: (B,T,16) -> (B,2,T,8)
```

Q/K 应用 RoPE 后，K/V 再沿 head 轴复用，与 4 个 query heads 对齐。从分数到最终输出的 shape 依次是：

```text
scores:  (B,4,T,T)
weights: (B,4,T,T)
heads:   (B,4,T,8)
concat:  (B,T,32)
output:  (B,T,32)
```

![Multi-head attention：各 head 合并后再做输出投影](assets/mha.png)

合并后的宽度回到 `D`，block 因而可以将 Attention 输出与原输入做残差相加。

## GQA 减少 K/V heads

标准 MHA 的 query、key 和 value head 数相同，Grouped-query attention（GQA）则保留 query heads 的数量，让一组 K/V 同时服务多个 query heads。在前面的 4 个 query heads、2 个 KV heads 配置中，对应关系是：

```text
n_heads = 4
n_kv_heads = 2
kv_repeats = 2

Q0,Q1 -> KV0
Q2,Q3 -> KV1
```

![4 个 query heads 共享 2 组 K/V](assets/gqa.png)

<div class="widget-mount" data-widget="gqa-map"></div>

每个 query head 仍然产生自己的输出，所以结果是 `O0..O3` 四个 head，不是两个。`_repeat_kv` 只在计算时展开 K/V，不会新建参数；原始 K/V 投影和后面的 KV Cache 都只有 2 个 heads。这个配置会将 K/V 缓存元素减半，却不会让整个模型的内存或耗时也恰好减半。

要让上述拆分、分组共享和 RoPE 旋转都能对齐，模型尺寸需要满足：

```text
D % n_heads == 0
n_heads % n_kv_heads == 0
head_dim % 2 == 0       # RoPE 成对旋转
```

## Causal mask 和 padding mask 是两件事

Causal mask 的 shape 可写成 `(1,1,T,T)`，它在 batch 和 head 维广播，每个样本共用同样的时间方向。`attention_mask: (B,T)` 则随样本而变，这份代码约定 `True` 表示该 key 有效，使用时扩展为：

```text
(B,T) -> (B,1,1,T)
```

两张 mask 取交集后得到最终的 allowed 矩阵。Padding mask 只屏蔽 key，不删除 PAD query 行，所以输出 shape 仍是 `(B,T,D)`；训练时还需要在 loss 端忽略对应的 PAD target。当某个 query 行连一个有效 key 都没有时，本实现会在 Softmax 后再乘一次 allowed mask，把整行权重置零。

## 前缀不应被未来改写

检查 causal 方向时，可以固定前四个位置，只替换右边的输入：

```python
x2 = x.clone()
x2[:, 4:] = torch.randn_like(x2[:, 4:])

y1 = attention(x)
y2 = attention(x2)
```

`y1[:, :4]` 与 `y2[:, :4]` 应在浮点容差内一致；反过来改动前文，后面位置的输出通常会随之改变。前一个现象排除未来泄漏，后一个现象则说明 Attention 确实读到了前文。文件末尾的小例子会打印整层输出和 head 配置：

```bash
python exercises/block_03_transformer/task_23_causal_attention/mha.py
```

```text
output: (2, 6, 32)
Q heads / KV heads: 4 / 2
```

Attention 完成了 token 之间的信息交换，[SwiGLU FFN](../task_24_swiglu_ffn/README.md) 则保持位置相互独立，继续在每个 token 内部变换特征。

参考：[GQA 论文](https://arxiv.org/abs/2305.13245)、[PyTorch `scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。
