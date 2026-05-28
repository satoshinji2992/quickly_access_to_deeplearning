# Block 3: Attention Is All You Need 到 MiniMind

第三块进入文本生成.

这块分两段走.

第一段先讲 Attention Is All You Need 里的基本想法: token 怎么互相看, 为什么需要位置编码, 为什么要多头, 为什么要 mask.

第二段再收束到 MiniMind/LLaMA 这一类 decoder-only 小模型: RoPE、GQA、RMSNorm、SwiGLU、next-token training、generate 和 KV cache.

Transformer 家族很大. 这一章只取和 MiniMind 主线关系最密的部分. Cross-attention、encoder-decoder、多模态和对齐训练放在拓展主题里, 读完主线后再看会更顺.

---

## 一. 先看原始 Transformer

Attention Is All You Need 做了几件关键的事:

- 用 self-attention 让 token 直接互相读取信息.
- 用 sinusoidal position encoding 给 token 加顺序.
- 用 multi-head attention 让模型从多个子空间看上下文.
- 在 decoder 里用 causal mask 防止偷看未来.
- 用 FFN 和残差堆出多层结构.

原论文里的 Transformer 是 encoder-decoder 结构. 但 MiniMind 主线只需要 decoder-only.

所以我们会讲原理, 但实现会往 decoder-only 收.

---

## 二. 再做 MiniMind

MiniMind 风格模型大概长这样:

```text
token -> embedding -> decoder blocks -> logits -> next token
```

每个 decoder block 里:

```text
RMSNorm -> Causal Attention + RoPE/GQA -> Residual
RMSNorm -> SwiGLU FFN -> Residual
```

你会用 PyTorch 写这些模块. 可以用 `nn.Linear`、`nn.Embedding` 这种基础层, 但不要直接调用一个现成大 Transformer 模块糊过去.

这里要看懂每个 shape.

---

## 任务安排

| 任务 | 问题 | 重点 |
| --- | --- | --- |
| [task_20](../exercises/block_03_transformer/task_20_transformer_theory/README.md) | Attention Is All You Need 到底发明了什么 | token、QKV、self-attention、mask |
| [task_21](../exercises/block_03_transformer/task_21_sinusoidal_position/README.md) | 原始 Transformer 怎么表示位置 | sinusoidal position encoding |
| [task_22](../exercises/block_03_transformer/task_22_rope_position/README.md) | MiniMind 为什么换成 RoPE | 旋转 Q/K、相对位置 |
| [task_23](../exercises/block_03_transformer/task_23_causal_attention/README.md) | token 怎么看前文 | causal self-attention、MHA、GQA |
| [task_24](../exercises/block_03_transformer/task_24_swiglu_ffn/README.md) | attention 后还要什么 | SwiGLU FFN |
| [task_25](../exercises/block_03_transformer/task_25_embedding_lm_head/README.md) | token 怎么进出模型 | embedding、LM head、weight tying |
| [task_26](../exercises/block_03_transformer/task_26_decoder_blocks/README.md) | block 怎么堆 | RMSNorm、残差、Pre-Norm |
| [task_27](../exercises/block_03_transformer/task_27_minimind_core/README.md) | MiniMind Core 怎么搭 | config、decoder-only、小模型 |
| [task_28](../exercises/block_03_transformer/task_28_next_token_training/README.md) | 怎么训练它 | next-token loss、toy training |
| [task_29](../exercises/block_03_transformer/task_29_generate_sampling/README.md) | 怎么让它生成 | temperature、top-k |
| [task_30](../exercises/block_03_transformer/task_30_kv_cache/README.md) | 推理怎么加速 | KV cache |

---

## 拓展主题

主线之外还有一些很常见的方向:

- Cross-Attention: encoder-decoder、多模态和扩散模型里经常出现.
- Tokenizer: BPE、SentencePiece、词表大小会直接影响训练和推理.
- SFT / DPO / RLHF: 让模型更像助手, 而不是只会续写.
- MoE / Mamba / State Space Model: Transformer 之外或 Transformer 旁边的架构路线.

---

## 做完以后

做完这一块, 你应该能说清楚:

- self-attention 为什么能让 token 互相读信息.
- sinusoidal position encoding 在解决什么.
- RoPE 和普通位置编码有什么不同.
- decoder-only 模型为什么需要 causal mask.
- MHA 和 GQA 的区别.
- RMSNorm 和 SwiGLU 在 MiniMind 里分别做什么.
- embedding、LM head、next-token loss 怎么连起来.
- generate 为什么一开始会慢, KV cache 又在缓存什么.
