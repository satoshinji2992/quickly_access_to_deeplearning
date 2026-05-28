# task_26: Decoder Block 叠加

现在你已经有 attention、FFN 和 embedding 了.

接下来要把它们组成 decoder block.

MiniMind/LLaMA 风格一般用 Pre-Norm:

```text
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

## 一. 为什么要残差?

Block 2 里 ResNet 已经讲过残差.

Transformer 也一样. 如果没有残差, 深层模型很难训练. 残差让信息和梯度都有一条更短的路.

## 二. 为什么是 Pre-Norm?

Post-Norm 是先做子层, 再归一化:

```text
x = Norm(x + Attention(x))
```

Pre-Norm 是先归一化, 再进子层:

```text
x = x + Attention(Norm(x))
```

现在很多 decoder-only 模型更偏向 Pre-Norm, 训练深层模型时更稳.

## 三. RMSNorm

MiniMind 主线用 RMSNorm.

RMSNorm 不减均值, 只按均方根缩放:

$$
\mathrm{RMSNorm}(x)=\frac{x}{\sqrt{\mathrm{mean}(x^2)+\epsilon}}\odot w
$$

它比 LayerNorm 少一点计算, 也很常见.

## 四. 你要写什么?

当前 `transformer_blocks.py` 先给了一个通用 block 版本.

主线目标是理解结构:

```text
Norm -> Attention -> Residual
Norm -> FFN -> Residual
```

下一关会把这些东西收进 `MiniMindCore`.
