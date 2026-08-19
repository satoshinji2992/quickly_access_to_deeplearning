# task_26：组装 Decoder Block

Attention 和 FFN 的输入输出都是 `(B,T,D)`。把它们各放进一条残差支路，就得到本章使用的 Pre-Norm decoder block：

```text
x1 = x  + Attention(RMSNorm1(x))
x2 = x1 + FFN(RMSNorm2(x1))
```

![Pre-RMSNorm decoder block 的两条残差路径](assets/decoder_block.png)

第二个 RMSNorm 读取 `x1`，而不是最初的 `x`。第一条支路已经更新表示，FFN 接着处理这份新结果。

---

## Pre-Norm 放在哪里

Pre-Norm 的通式是：

```text
x + sublayer(norm(x))
```

Post-Norm 则是：

```text
norm(x + sublayer(x))
```

两者都是合法设计。本章的结构图、公式和代码统一采用 Pre-RMSNorm；每层有两个独立 norm，全部 block 之后还有一次 final RMSNorm。

残差相加成立的前提是 sublayer 输出与输入 shape 相同：

$$
x_1=x+\operatorname{Attention}(\operatorname{RMSNorm}_1(x)),
$$

$$
x_2=x_1+\operatorname{FFN}(\operatorname{RMSNorm}_2(x_1)).
$$

若 FFN 忘了从 `hidden_dim` 投影回 `D`，错误会在第二次加法处直接暴露。

## RMSNorm 与 LayerNorm 的区别

这里的 RMSNorm 是：

$$
\operatorname{RMSNorm}(x)
=\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}\odot w.
$$

它只沿最后一维计算均方根并学习缩放 `w`，不减均值，也没有 bias。LayerNorm 通常先减均值，再按方差缩放；名称不能混用。

Shape 约定为：

```text
input:  (...,D)
weight: (D,)
output: (...,D)
```

输入 `(B,T,D)` 时，每个 token 独立在最后一维归一化，不混合 batch 或 sequence。RMSNorm 没有 running statistics，因此 train/eval 使用相同公式。代码对 FP16/BF16 先用 FP32 计算均方根，再转回输入 dtype，避免小精度下不必要的数值损失。

## 可注入的 attention 和 FFN

`TransformerBlock` 的构造函数接收两个现成模块：

```python
block = TransformerBlock(
    dim,
    attention=my_attention,
    feed_forward=my_ffn,
)
```

这样可以把两个 norm、两个 sublayer 和两次 residual 的组装与 attention 内部实现分开观察。脚本底部用 `nn.Linear` 与一个小 `nn.Sequential` 做 smoke test；它没有 RoPE、GQA 或 causal mask。

Task 27 的 `DecoderBlock` 才固定接入 `CausalSelfAttention` 和 `SwiGLU`。因此，这里的脚本成功运行只表明 block 组装方式正常，不包含对 attention 性质的结论。

## 堆叠 block

`TransformerStack` 用 `nn.ModuleList` 保存多个 block：

```python
for block in self.blocks:
    x = block(x)
```

每层都保持 `(B,T,D)`，参数默认不共享。列表中是多个不同的 block 对象；如果重复放入同一实例，就会变成跨层权重共享。

## 运行与核对

```bash
python exercises/block_03_transformer/task_26_decoder_blocks/transformer_blocks.py
```

输出形如：

```text
stack output: (2, 5, 16)
normalization: RMSNorm
```

Block 组装性质也收录在 Block 3 测试中：

```bash
python -m unittest discover -s tests -p 'test_block3.py' -v
```

运行结果与测试覆盖 RMSNorm 的最后一维计算、`norm1` 与 `norm2` 的独立性、两条 residual 的输入来源，以及 stack 前后的 shape 和反向传播。Causal、RoPE 与 GQA 在 task 27 的完整 block 中统一观察。

参考：[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)。
