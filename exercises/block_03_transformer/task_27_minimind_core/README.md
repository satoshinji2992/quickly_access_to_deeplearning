# 拼成 MiniMind Core

Embedding、Attention、SwiGLU 和 decoder block 分开时容易看清公式，语言模型却必须把它们接成一条路。`minimind_core.py` 实现了一个小型 decoder-only Transformer：输入是 token ids，输出是每个位置对整个词表的 logits。

![MiniMind 的模型主干、训练目标和缓存推理](assets/minimind_overview.png)

## 从 token ids 到 logits

```text
input_ids (B,T)
  -> token embedding (B,T,D)
  -> N × DecoderBlock
       u = x + CausalRoPEGQA(RMSNorm(x))
       x = u + SwiGLU(RMSNorm(u))
  -> final RMSNorm
  -> tied LM head
  -> logits (B,T,V)
```

模型不使用 learned position embedding，每层 Attention 都会对自己的 Q/K 应用 RoPE。Attention 负责读取前缀，SwiGLU 逐位置处理特征，两条残差支路都回到模型宽度，因而每个 block 前后都保持 `(B,T,D)`。

这些部件的尺寸都收在 `MiniMindConfig` 中，不再分散到各层内部：

| 字段 | 含义 |
| --- | --- |
| `vocab_size` | 词表大小 `V` |
| `dim` | 模型宽度 `D` |
| `n_layers` | DecoderBlock 数量 |
| `n_heads` | query heads 数量 |
| `n_kv_heads` | K/V heads 数量 |
| `hidden_dim` | SwiGLU 中间宽度 |
| `max_seq_len` | 最大可见窗口 |
| `rope_base` | RoPE 频率底数 |
| `norm_eps` | RMSNorm 稳定项 |
| `pad_token_id` | PAD id；没有 PAD 时可设为 `None` |

`dim` 要能被 `n_heads` 整除，否则特征无法均分给各 query head；`n_heads` 又要能被 `n_kv_heads` 整除，才能让每组 K/V 服务相同数量的 query heads。RoPE 还需要两维一对地旋转，所以 `head_dim = dim / n_heads` 必须为偶数。配置对象会在模型创建前检查这些条件。

task 22–26 中的 Python 文件都可以独立运行。为了避免跨任务目录的动态导入，`minimind_core.py` 没有直接 import 那些示例类，而是把同一组公式整合成支持 `attention_mask` 和 cache 接口的版本。因此，前几页是独立可读的部件实现，不是整模在运行时直接导入的依赖。

## 一层 Attention 的 shape 与可见范围

对输入 `x: (B,T,D)`，一层 Attention 会先投影并拆分 head，得到：

```text
Q: (B,n_heads,T,head_dim)
K: (B,n_kv_heads,T,head_dim)
V: (B,n_kv_heads,T,head_dim)
```

Q/K 经过 RoPE，K/V 按组展开到 `n_heads`，score 再受 causal mask 和 padding mask 共同限制。各 head 汇总的结果拼回 `(B,T,D)` 后，才能进入残差加法。替换未来 token 不应改写前缀，这个性质可以用两行输入直接比较：

```text
[1, 2, 3, 4, 5]
[1, 2, 3, 9, 8]
```

前 3 个位置相同，所以它们的 logits 不会被后面的 9、8 改写。把改动移到前文后：

```text
[1,  2,  3, 4, 5]
[1, 11, 12, 4, 5]
```

改动 11、12 后，最后位置的 logits 通常也会变化。前一组对照 causal 方向，后一组说明最后位置确实读到了前文。

## 第二条残差读取第一条的结果

两条残差有明确的先后依赖，代码顺序为：

```python
attn_out = attn(attn_norm(x))
x = x + attn_out

ffn_out = ffn(ffn_norm(x))
x = x + ffn_out
```

`ffn_norm` 读取的是已经加入 Attention 输出的 `x`，两次加法不能并成一次。每层有两个独立 RMSNorm，SwiGLU 也分别使用 `gate_proj`、`up_proj` 和 `down_proj`。

所有 blocks 处理完后，整模做一次 final RMSNorm，然后进入与 token embedding 共享权重的 LM head：

```python
self.lm_head.weight = self.token_embedding.weight
```

两个模块始终引用同一个 Parameter，所以输入查表与输出打分产生的梯度会累积到同一张表上。

## Forward 的基本接口

只需要 logits 时，最简单的调用是：

```python
logits, loss = model(input_ids)
```

`input_ids` 是 `(B,T)`，`logits` 是 `(B,T,V)`，未传入 labels 时 `loss` 为 `None`。序列长度不能超过 `max_seq_len`；带 PAD 的 batch 可以再传入 `(B,T)` 的 `attention_mask`。

本页只看基本 forward。Labels 和数据切片放在 [Next-token 训练](../task_28_next_token_training/README.md) 中，cache 参数则到 [KV Cache](../task_30_kv_cache/README.md) 再解释。直接运行模型文件，可以看到输出 shape 和这个配置的真实参数量：

```bash
python exercises/block_03_transformer/task_27_minimind_core/minimind_core.py
```

```text
logits: (1, 4, 32)
parameters: 19616
```

`logits` 的三个轴分别对应 batch、sequence 和 vocabulary。[Next-token 训练](../task_28_next_token_training/README.md) 会把原始文本变成 `input_ids`、labels 和 masks，再把 logits 接到损失计算上。
