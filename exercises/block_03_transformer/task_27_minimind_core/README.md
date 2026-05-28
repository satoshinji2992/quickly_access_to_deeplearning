# task_27: MiniMind Core

这一关把前面的东西收起来.

目标是搭一个 MiniMind 风格的 decoder-only 小模型.

它不一定大, 但结构要对.

## 一. 配置类

先用一个 config 管住模型规模:

```text
vocab_size
dim
n_layers
n_heads
n_kv_heads
hidden_dim
max_seq_len
```

这样你不用在代码里到处写数字.

## 二. 模型结构

主线结构是:

```text
token_embedding
decoder blocks
final RMSNorm
lm_head
```

每个 block 里:

```text
RMSNorm -> Causal Attention with RoPE/GQA -> Residual
RMSNorm -> SwiGLU -> Residual
```

当前 starter 里先给了一个教学骨架. 它能跑通 next-token loss, 但 attention 部分还需要你逐步替换进去.

完整参考可以看 `solutions/block_03_transformer/minimind_solution.py`.

## 三. 检查什么?

- 输入 `input_ids` shape: `(batch, seq_len)`.
- 输出 logits shape: `(batch, seq_len, vocab_size)`.
- labels 不为空时能返回 loss.
- `parameter_count()` 能打印参数量.
- `seq_len` 不应该超过 `max_seq_len`.

下一关训练它.
