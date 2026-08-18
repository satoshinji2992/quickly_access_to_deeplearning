# 从 Attention 到微型语言模型

Block 3 围绕一个能按上下文预测下一个 token 的 PyTorch 小模型展开，再向外连接训练、采样和缓存推理。

这里的“微型”是认真的。默认模型只有两层，语料是一小段内置中文，tokenizer 也只是字符表。它适合拆开检查 Transformer 的数据流，不足以训练出通用对话模型。

![Block 3 概览：因果注意力、RoPE、GQA 与 KV Cache](../assets/images/block3_attention_overview.png)

---

## Decoder-only Transformer 在算什么

给定 hidden states `X: (B,T,D)`，self-attention 先投影出 Q、K、V。单个 head 的计算是：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_{head}}}+M\right)V.
$$

`QKᵀ` 决定每个位置从哪里取信息，`V` 提供被取回的内容。$M$ 是 mask：decoder-only 模型用下三角 causal mask，位置 $t$ 只能读取 `0..t`。因此训练时虽然整段序列一次进入模型，前面的表示仍看不到后面的答案。

原始 Transformer 是 encoder-decoder 架构。本章只实现 decoder：没有 encoder，也没有 cross-attention，Q/K/V 都来自同一段序列。Task 20 先对比这两种结构，后面再进入代码。

Attention 本身没有顺序概念。Task 21 用原论文的正弦位置编码建立频率直觉；Task 22 改为本章主线使用的 RoPE。RoPE 不把位置向量加到 embedding，而是旋转 Q/K。每对维度使用不同频率，旋转后的 Q/K 点积会显式依赖位置差 $n-m$。

Task 23 再加入 GQA。若 `n_heads=8, n_kv_heads=2`，每 4 个 query heads 共享一组 K/V；attention 仍产生 8 个 query-head 输出，只是 K/V 投影和缓存少了。

## 一层 Decoder Block

![Pre-RMSNorm Decoder Block](../assets/images/decoder_block.png)

本章统一使用 Pre-RMSNorm：

```text
x = x + CausalRoPEGQA(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

两条支路最后都回到 `(B,T,D)`，所以能与输入相加。SwiGLU 先做两次独立的上投影：

$$
\operatorname{SwiGLU}(x)
=W_{down}\bigl(\operatorname{SiLU}(W_{gate}x)\odot W_{up}x\bigr).
$$

完整模型的数据流如下：

```text
input_ids (B,T)
  -> token embedding
  -> N × decoder block
  -> final RMSNorm
  -> LM head
  -> logits (B,T,V)
```

Embedding 与 LM head 共用同一个 `(V,D)` 参数矩阵。代码用对象身份确认两处引用的是同一参数：

```python
model.lm_head.weight is model.token_embedding.weight
```

## 训练目标和两种 mask

Next-token 训练把同一段 token 错开一位：

```text
tokens: [BOS, t0, t1, t2, EOS, PAD]
input:  [BOS, t0, t1, t2, EOS]
label:  [t0,  t1, t2, EOS, PAD]
```

![Next-token 标签错位与 mask](../assets/images/shifted_labels.png)

Padding 牵涉两处，不能只处理 loss：

- attention mask 屏蔽 PAD key，防止有效 token 读取填充值；
- loss mask 把 PAD target 变成 `-100`，交给 cross-entropy 忽略。

Task 28 先切分 train/validation 文本，再只用训练文本建立字符词表。训练脚本保存模型配置、参数、优化器、tokenizer、step 和 validation loss。它可以验证数据、反向传播和 checkpoint 是否接通，但一次有限的 validation loss 并不代表模型已经有语言能力。

```bash
python exercises/block_03_transformer/task_28_next_token_training/train.py \
  --steps 80 --checkpoint /tmp/minimind_demo.pt
```

## 从 logits 到连续生成

生成时只使用最后位置的 `logits[:, -1]`。Task 29 实现四种选择方式：

- greedy：直接取最大 logit；
- temperature：改变概率分布的尖锐程度；
- top-k：只在最高的一小组候选中采样；
- top-p：保留累计概率达到阈值的最小候选集。

未缓存生成每次都会重算当前窗口。Task 30 保存每一层尚未展开的 K/V：

```text
K, V: (B, n_kv_heads, past_len, head_dim)
```

新 token 仍要计算自己的 Q/K/V，也仍要让新 Q 与所有可见 K 做 attention；省掉的是旧 token 的层计算和 K/V 投影。RoPE 的位置从 `past_len` 接着走，mask 也要覆盖缓存前缀与当前 token。实现是否正确，最终由 cached/full logits 的数值误差和 greedy token 序列共同判断。

```bash
python exercises/block_03_transformer/task_30_kv_cache/kv_cache.py \
  --checkpoint /tmp/minimind_demo.pt --prompt "清晨，"
```

## 章节索引

| 小节 | 内容 | 可观察的性质 |
| --- | --- | --- |
| [task_20](../exercises/block_03_transformer/task_20_transformer_theory/README.md) | Attention 与 decoder-only | Q/K/V 来源、causal mask |
| [task_21](../exercises/block_03_transformer/task_21_sinusoidal_position/README.md) | 正弦位置编码 | shape、第 0 行、偶数维约束 |
| [task_22](../exercises/block_03_transformer/task_22_rope_position/README.md) | RoPE | 分维度频率、相对位移、`start_pos` |
| [task_23](../exercises/block_03_transformer/task_23_causal_attention/README.md) | Causal GQA | 未来不影响过去，KV heads 确实减少 |
| [task_24](../exercises/block_03_transformer/task_24_swiglu_ffn/README.md) | SwiGLU | gate/up 双分支，shape 保持 |
| [task_25](../exercises/block_03_transformer/task_25_embedding_lm_head/README.md) | Embedding 与 LM head | weight tying、PAD loss |
| [task_26](../exercises/block_03_transformer/task_26_decoder_blocks/README.md) | Decoder block | Pre-RMSNorm、两条 residual |
| [task_27](../exercises/block_03_transformer/task_27_minimind_core/README.md) | MiniMind Core | causal、上下文敏感、梯度与 cache shape |
| [task_28](../exercises/block_03_transformer/task_28_next_token_training/README.md) | 训练闭环 | train/val 隔离、过拟合小 batch、checkpoint round-trip |
| [task_29](../exercises/block_03_transformer/task_29_generate_sampling/README.md) | 自回归采样 | greedy、temperature、top-k、top-p |
| [task_30](../exercises/block_03_transformer/task_30_kv_cache/README.md) | 缓存推理 | cached/full 等价 |

上述性质都集中在 Block 3 的回归测试中：

```bash
python -m unittest discover -s tests -p 'test_block3.py' -v
```

## 参考

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [PyTorch `scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Hugging Face：Caching](https://huggingface.co/docs/transformers/main/cache_explanation)
