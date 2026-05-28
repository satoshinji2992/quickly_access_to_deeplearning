# task_20: Attention Is All You Need 到 MiniMind

这一关先不写复杂代码.

你要先知道 Transformer 到底在解决什么问题, 以及 MiniMind 为什么只取其中一部分.

## 一. RNN 的麻烦

在 Transformer 之前, 很多序列模型靠 RNN 一步一步读 token.

```text
t0 -> t1 -> t2 -> t3
```

这样很自然, 但也有问题:

- 不能很好并行.
- 长距离信息传起来费劲.
- 很多步以后, 前面的信息容易变淡.

Attention 的想法更直接: 每个 token 可以直接看别的 token.

## 二. Q、K、V

Self-attention 会把每个 token 的 hidden state 投影成三份:

```text
Q: 我想找什么
K: 我有什么特征可以被匹配
V: 如果你看我, 我给你什么信息
```

然后用:

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

这就是 attention 的核心公式.

除以 $\sqrt{d_k}$ 是为了控制数值尺度. 不除的话, head_dim 大时 score 容易变得很大, softmax 会太尖.

## 三. 位置从哪里来?

Attention 本身不关心顺序.

如果不给位置, 它看到的更像一袋 token.

所以原始 Transformer 用 sinusoidal position encoding, 把“第几个 token”加进 embedding.

后面的 MiniMind 主线会改用 RoPE. 但先理解 sinusoidal, 再看 RoPE 会顺很多.

## 四. encoder-decoder 和 decoder-only

Attention Is All You Need 原论文是 encoder-decoder.

翻译任务里, encoder 读源语言, decoder 生成目标语言. decoder 还会用 cross-attention 去读 encoder 输出.

但 MiniMind/GPT/LLaMA 是 decoder-only.

它只做一件事:

```text
看前面的 token, 预测下一个 token
```

所以主线会重点实现 causal self-attention. Cross-attention 属于 encoder-decoder 和多模态方向, 可以作为拓展主题单独看.

## 五. 接下来怎么走?

先看原始位置编码, 再看 RoPE.

然后写 causal attention, 接 SwiGLU、RMSNorm、embedding 和 LM head, 最后搭 MiniMind Core.
