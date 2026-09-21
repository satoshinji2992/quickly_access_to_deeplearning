# 组装 Decoder Block

[Causal Attention](../task_23_causal_attention/README.md) 读取前缀，[SwiGLU](../task_24_swiglu_ffn/README.md) 在每个 token 内部变换特征。两者都从 `(B,T,D)` 输入，也都输出 `(B,T,D)`，恰好可以各自放进一条残差支路：

```text
u = x + Attention(RMSNorm1(x))
h = u + FFN(RMSNorm2(u))
```

![Pre-RMSNorm decoder block 的两条残差路径](assets/decoder_block.png)

第二个 RMSNorm 读取的是已经加入 Attention 输出的 `u`，不是最初的 `x`，所以刚刚写入的上下文信息会继续进入 FFN。

## 残差连接保留了直接路径

若只把很多子层首尾相接，输入和梯度每经过一层都必须穿过该层的完整变换。改成 `x + sublayer(...)` 之后，即使子层暂时没学到有用改动，原向量仍能继续向后传递，反向梯度也多了一条加法路径。这个相加操作要求主路与子层输出的 shape 完全相同：

$$
u=x+\operatorname{Attention}(\operatorname{RMSNorm}_1(x)),
$$

$$
h=u+\operatorname{FFN}(\operatorname{RMSNorm}_2(u)).
$$

这正是 Attention 在各 head 拼接后还要投影回 `D`、FFN 也要从 `hidden_dim` 投影回 `D` 的原因。

## Pre-Norm 先归一化，再进子层

Pre-Norm 和 Post-Norm 使用的子层可以相同，差别只在归一化放在残差相加之前还是之后：

```text
Pre-Norm:   x + sublayer(norm(x))
Post-Norm:  norm(x + sublayer(x))
```

本仓库使用 Pre-RMSNorm，进入 Attention 和 FFN 之前各有一个独立 RMSNorm，残差主路上的 `x` 不会被这次归一化改写。`transformer_blocks.py` 的范围只到 block 和 stack，因而不在 stack 末尾加 final norm；完整的 `MiniMindCore` 会在所有 blocks 之后另加 final RMSNorm，再进入 LM head。

## RMSNorm 不减均值

对一个 token 的特征向量 $x$，RMSNorm 先用均方根缩放整个向量，再乘以可学习权重 $w$：

$$
\operatorname{RMSNorm}(x)
=\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}\odot w.
$$

以 `[3,4]` 为例，均方根是 $\sqrt{(9+16)/2}\approx3.5355$。忽略很小的 $\epsilon$，并令 $w=[1,1]$，输出约为 `[0.8485,1.1314]`，均值并不是 0。LayerNorm 通常会先减均值，再除以标准差（更准确地说是 $\sqrt{\mathrm{variance}+\epsilon}$）；RMSNorm 不减均值，它使用的是均方根。

当输入是 `(B,T,D)` 时，每个 token 都有自己的缩放因子，统计只沿最后一维计算：

```text
input:  (...,D)
weight: (D,)
output: (...,D)
```

这样不会混合 batch 和 sequence 维的统计。RMSNorm 没有 running statistics，训练和评估阶段使用同一公式；对 FP16/BF16 输入，代码会先转到 FP32 计算均方根，再转回原 dtype。

## 小脚本只演示 block 的连接方式

`TransformerBlock` 的构造函数接收现成的 attention 和 FFN 模块，所以同一个骨架可以搭配不同的子层实现：

```python
block = TransformerBlock(
    dim,
    attention=my_attention,
    feed_forward=my_ffn,
)
```

文件末尾用 `nn.Linear` 和一个小 `nn.Sequential` 代替真实 Attention/FFN，因而只呈现两个 norm、两条残差与 block 的 shape 流转；RoPE、GQA 和 causal mask 会在 [MiniMind Core](../task_27_minimind_core/README.md) 的完整 block 中接入。

多层结构由 `TransformerStack` 使用 `nn.ModuleList` 保存，forward 时依次通过每一层：

```python
for block in self.blocks:
    x = block(x)
```

每层都保持 `(B,T,D)`，层与层之间默认使用独立参数；只有在列表里重复放入同一个 block 实例，才会形成跨层权重共享。运行该文件可以看到 stack 的输出 shape 和归一化类型：

```bash
python exercises/block_03_transformer/task_26_decoder_blocks/transformer_blocks.py
```

```text
stack output: (2, 5, 16)
normalization: RMSNorm
```

这个 block 骨架与真实 Attention、SwiGLU、Embedding、final RMSNorm 和 LM head 接起来后，就得到完整的 [MiniMind Core](../task_27_minimind_core/README.md)。

参考：[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)。
