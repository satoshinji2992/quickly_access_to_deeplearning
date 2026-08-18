# task_27：MiniMind Core

前面几份代码各自演示一个部件。`minimind_core.py` 把它们接成一台可以训练的微型 decoder-only Transformer。

这里说的“模型主干齐全”只指数据流：RoPE、causal GQA、RMSNorm、SwiGLU、残差和 weight tying 都参与 forward。它仍是小型教学模型，参数量和训练数据都不能与通用 LLM 相提并论。

![MiniMind 的模型主干、训练目标和缓存推理](assets/minimind_overview.png)

---

## 从 token ids 到 logits

```text
input_ids (B,T)
  -> token embedding (B,T,D)
  -> N × DecoderBlock
       x = x + CausalRoPEGQA(RMSNorm(x))
       x = x + SwiGLU(RMSNorm(x))
  -> final RMSNorm
  -> tied LM head
  -> logits (B,T,V)
```

模型没有 learned position embedding。每层 attention 都会对自己的 Q/K 应用 RoPE。

若传入已经错位的 `labels: (B,T)`，forward 额外返回一个 masked cross-entropy loss；若 `use_cache=True`，还会返回每层的新 K/V。

`MiniMindConfig` 控制以下尺寸：

| 字段 | 含义 |
| --- | --- |
| `vocab_size` | 词表大小 `V` |
| `dim` | 模型维度 `D` |
| `n_layers` | DecoderBlock 数量 |
| `n_heads` | query heads 数量 |
| `n_kv_heads` | K/V heads 数量 |
| `hidden_dim` | SwiGLU 中间宽度 |
| `max_seq_len` | 最大可见窗口 |
| `rope_base` | RoPE 的频率底数 |
| `norm_eps` | RMSNorm 稳定项 |
| `pad_token_id` | PAD id；不需要时可设为 `None` |

配置创建时就会检查正整数、head 整除关系、偶数 `head_dim`、有限的 `rope_base/norm_eps` 以及 PAD id 范围。这样错误不会拖到第一次 `view` 或矩阵乘法才出现。

## Attention 确实读取上下文

以一层为例：

```text
x: (B,T,D)

Q: (B,n_heads,T,head_dim)
K: (B,n_kv_heads,T,head_dim)
V: (B,n_kv_heads,T,head_dim)
```

Q/K 经过 RoPE；K/V 按组展开到 `n_heads`；score 叠加 causal/padding mask；各 head 的结果合回 `(B,T,D)`。因此修改 `n_heads` 或 `n_kv_heads` 会真实改变投影和 cache shape，不是只改配置文件里的名字。

上下文连接可以从两个互补的现象中观察。

**未来不影响过去**：

```text
[1,2,3,4,5]
[1,2,3,9,8]
```

前 3 个位置的 logits 在容差内一致。

**过去会影响后来**：

```text
[1, 2, 3,4,5]
[1,11,12,4,5]
```

最后位置的 logits 通常会不同。差异恒为 0 时，常见原因是 attention 被写成了逐 token MLP，或 residual 绕开了 attention 输出。

反向传播后，`blocks[0].attn.q_proj.weight.grad` 存在且含有非零元素，说明 attention 路径确实参与了损失计算。

## DecoderBlock 与共享输出权重

每层的两个 RMSNorm 和两条残差是独立的：

```python
attn_out = attn(attn_norm(x))
x = x + attn_out

ffn_out = ffn(ffn_norm(x))
x = x + ffn_out
```

SwiGLU 包含 `gate_proj`、`up_proj` 和 `down_proj` 三份无 bias 投影。

模型初始化后把 LM head 绑到 token embedding：

```python
self.lm_head.weight = self.token_embedding.weight
```

对象身份可以直接确认这个共享关系：

```python
model.lm_head.weight is model.token_embedding.weight
```

新模型先建立这层引用，再载入 `state_dict`，所以 checkpoint round-trip 后共享关系仍在。

## Padding 与 labels 的接口边界

未显式传 `attention_mask` 时，模型根据 `pad_token_id` 生成：

```python
attention_mask = input_ids.ne(pad_token_id)
```

Attention 用它屏蔽 PAD key；loss 还会忽略 label 为 PAD 或 query mask 为 False 的位置。若没有任何有效 label，返回与 logits 相连的 0 loss。

模型不会自动 shift labels。这个接口接收 next-token Dataset 产生的错位标签，Task 28 中可以看到完整数据路径。

带 padding 的变长生成 batch 采用左 padding，每行最后一位是有效 token，因为生成循环统一取 `logits[:, -1]`。缓存解码同时保留完整 attention mask；只传当前一步的 mask 会使 core 把历史前缀视为全有效，丢失 prompt 中原有的 PAD 信息。

## Cache 接口和慢速参考生成

普通 forward：

```python
logits, loss = model(input_ids, labels=None, attention_mask=None)
```

缓存 forward：

```python
logits, loss, new_past = model(
    input_ids,
    attention_mask=attention_mask,
    past_key_values=past,
    use_cache=True,
)
```

`new_past` 含每层一对 K/V：

```text
list[(K,V)]
K,V: (B,n_kv_heads,total_len,head_dim)
```

`MiniMindCore.generate` 是未缓存参考实现。它支持 greedy、temperature、top-k、top-p、EOS 与 mask，但每一步都会重新计算最近 `max_seq_len` 个 token。Task 30 会复用相同采样规则，改为复用逐层 cache；保留慢版本是为了比较 logits 和 token 序列。

## 运行与核对

```bash
python exercises/block_03_transformer/task_27_minimind_core/minimind_core.py
```

预期类似：

```text
logits: (1, 4, 32)
parameters: <正整数>
```

参数量会随配置变化。三个 logits 维度分别是 batch、sequence、vocabulary。

相关性质集中在 Block 3 测试中：

```bash
python -m unittest discover -s tests -p 'test_block3.py' -v
```

测试覆盖 shape、causal 与上下文敏感性、K/V heads 的实际数量、weight tying、PAD key/loss mask、attention 梯度，以及 cache 的层数与 shape。这些性质把“forward 能返回 tensor”进一步细化为可观察的模型行为，也为下一节解读 loss 变化提供基础。
