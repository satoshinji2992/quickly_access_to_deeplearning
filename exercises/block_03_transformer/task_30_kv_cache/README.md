# KV Cache

[未缓存生成](../task_29_generate_sampling/README.md) 每轮都会重新计算整个可见窗口。假设 prompt 有 5 个 token，虽然每轮只多生成一个 token，送入 forward 的长度却会持续增长：

```text
第 1 步：forward 5 tokens
第 2 步：forward 6 tokens
第 3 步：forward 7 tokens
...
```

参数固定、训练期随机操作关闭且前缀不变时，causal Attention 中的旧位置不会被新 token 改写，所以各层已经算好的 K/V 可以保存下来。新一轮只投影新 token 的 Q/K/V，再让新 Q 读取“历史 K/V + 当前 K/V”，这就是 KV Cache 避免重复计算的核心。

![每层 KV Cache：prefill 后逐 token 追加](assets/kv_cache.png)

<div class="widget-mount" data-widget="kv-cache"></div>

KV Cache 的使用场景是自回归推理。训练时所有位置已经在同一个张量中，仍然使用并行的完整 forward，不需要把训练也拆成逐 token 循环。

## Cache 按层保存 K/V

每个 decoder layer 都有自己的 K/V 投影，不同层的中间结果不能混在一起，因而 cache 按层保存为一个列表：

```text
past_key_values = [
    (K_layer0, V_layer0),
    (K_layer1, V_layer1),
    ...
]
```

列表中每个 K/V 张量都保留 batch、KV head、已缓存序列和 head 特征四个维度：

```text
(B,n_kv_heads,past_len,head_dim)
```

缓存中存的是尚未 `repeat_kv` 的 K/V。若模型有 8 个 query heads、2 个 KV heads，cache 只保留 2 份 K/V，计算 Attention 时才按组展开到 8 份，这也是 GQA 能够直接减少缓存体积的原因。

缓存并不会让单步 Attention 变成常数时间，因为新 query 仍要与所有可见 keys 计算分数。`repeat_kv` 之后，单轮解码的 shape 是：

```text
Q:       (B,n_heads,1,head_dim)
K:       (B,n_heads,past_len+1,head_dim)  # repeat_kv 后
scores:  (B,n_heads,1,past_len+1)
```

它省掉的是旧 token 重复经过各个 blocks 与 K/V 投影的计算，代价则是 cache 内存会随层数和可见前缀长度一起增长。

设 prompt 长度为 `P`，要生成 `S` 个 token，暂不考虑窗口截断，两条路径的计算账本可以写成：

| 路径 | 穿过整个 block stack 的序列长度 | 每个新 query 要读的 keys |
| --- | --- | --- |
| 未缓存 | `P, P+1, ..., P+S-1` | 当前全部前缀 |
| KV Cache | 先 prefill `P` 一次，之后每轮只输入 `1` | 仍然是当前全部前缀 |

缓存免去了旧 token 在 block 中的重复变换，但 Attention 的扫描长度仍会随前缀增长。若有 `L` 层，K/V cache 大约保存 `2 × L × B × n_kv_heads × (P+s) × head_dim` 个元素，前面的 2 分别对应 K 和 V。

## Prefill 处理 prompt，decode_one 处理新 token

正常生成开始时尚无缓存，`prefill` 会一次读取完整 prompt：

```python
logits, cache = prefill(model, input_ids, attention_mask)
```

它返回所有 prompt 位置的 logits 和每层 K/V。生成第一个新 token 只使用 `logits[:, -1]`，完整 logits 则保留给数值对照。新 token 选出后，`decode_one` 就只需接收 `(B,1)` 的 id：

```python
step_logits, cache = decode_one(
    model,
    next_id,             # (B,1)
    past_key_values,
    attention_mask,
)
```

各层会将新 K/V 沿序列轴追加，使缓存长度从 `past_len` 增加到 `past_len+1`。与之一起传入的 mask 必须同时覆盖缓存前缀和当前 token：

```text
attention_mask: (B,past_len+1)
```

Prompt 含左侧 PAD 时，原 mask 不能丢弃，每轮还要为新 token 追加一个 `True`。若省略 mask，`decode_one` 会把整个历史前缀都当成有效 token，原来的 PAD 也会被读取。

## RoPE 位置从 `past_len` 继续

缓存中已有 `past_len` 个位置时，新 token 并不是又从位置 0 开始，它的索引应当恰好是 `past_len`：

```python
cos, sin = build_rope_cache(
    seq_len=1,
    head_dim=head_dim,
    start_pos=past_len,
)
```

若每步都从位置 0 开始，代码仍能运行，cache 长度也会正常增长，但 Q/K 的旋转角已经与完整 forward 不一致。由于所有 shape 看起来都合法，这类错误需要靠数值对照发现。

位置偏移也决定了 causal mask 的形状。比如 cache 已有 3 个 token，这次一起追加 2 个，新 query 的位置就是 3 和 4，key 则涵盖 0 到 4：

```text
              k0 k1 k2 k3 k4
q3             1  1  1  1  0
q4             1  1  1  1  1
```

这是一个 `(2,5)` 的非方阵。若直接对它调用没有偏移的 `tril`，第一行只会留下 `k0`，缓存虽然存在却读不全。代码按 `key_position <= query_position` 生成 mask，让 query 的编号从 `past_len` 开始；每次只追加一个 token 时，它自然可以读取整个前缀以及自己。

## 完整 forward 是缓存实现的数值参考

检查缓存实现时，需要为整段输入拼出每个位置的 cached logits。`logits_with_kv_cache` 因此故意只用第一个 token 建立初始 cache，再逐 token 调用 `decode_one`。这是为了和完整 forward 逐位置对照的测试路径，不是上一节“整段 prompt 一次 prefill”的生成路径：

```python
full_logits, _ = model(input_ids, attention_mask=mask)
cached_logits = logits_with_kv_cache(model, input_ids, mask)
error = (full_logits - cached_logits).abs().max()
```

CPU float32 下，仓库固定的小模型样例的最大误差在 `1e-6` 以内；训练后的权重和更长的输入可能将绝对误差放大到数个 `1e-6`。比较时可同时使用绝对与相对容差，例如 `torch.testing.assert_close(cached_logits, full_logits, atol=1e-6, rtol=1e-5)`，不必要求所有浮点尾数相同。仓库的固定测试样例还会比较 greedy token ids，因为该样例中最高两项的差距足够大。换成其他模型或输入后，若两个候选几乎并列，微小浮点误差仍可能改变 argmax。

这次 cached/full 对照会经过每层 cache、K/V 追加、RoPE 位置偏移、非方阵 causal mask 和 padding mask。它能捕捉多种常见错位，但仍然只是针对给定输入的数值检查。

## 缓存生成的时序

```text
prompt -> prefill -> last logits + cache

repeat:
    sample next_id
    append next_id
    decode_one(next_id, cache)
```

这个循环只替换了模型 forward 的节奏，Temperature、top-k、top-p、EOS 和 batch 结束规则都沿用 [自回归生成与采样](../task_29_generate_sampling/README.md) 中的规则。在仓库固定的 greedy 测试中，缓存与未缓存版本产生相同的 token ids；对任意输入，仍应先看 logits 容差。

## 窗口滚动时重新 prefill

本仓库的参考生成只使用最近 `max_seq_len` 个 token。窗口已满又产生新 token 时，代码会截取最近的窗口，并把其中第一个 token 重新视为位置 0：

```python
visible = result[:, -max_seq_len:]
```

直接删掉最早的 K/V 还不够。当前接口用 cache 长度推导新 token 的 RoPE 位置；裁短 cache 后，新 Q 会按缩短后的长度编号，保留下来的 K 却仍带着裁剪前的旋转位置，两边不再处在同一套坐标中。即使另存绝对位置计数器，更深层的 K/V 也已经混入被移出窗口的旧上下文，不能等同于只用新窗口重新计算的结果。

为了与未缓存版本“截取窗口后从头 forward”的语义严格一致，缓存版在窗口滚动时重新 prefill。这样会在滚动发生的那一轮暂时失去增量计算的优势，但两条路径仍可直接做数值对照。

## 从 checkpoint 启动缓存生成

Checkpoint 由 [Next-token 训练](../task_28_next_token_training/README.md) 脚本产生。不传采样参数时，缓存脚本使用 greedy 生成：

```bash
python exercises/block_03_transformer/task_30_kv_cache/kv_cache.py \
  --checkpoint /tmp/minimind_demo.pt --prompt "周一早晨，" \
  --max-new-tokens 20
```

脚本会先打印 cached/full 的最大 logits 误差，然后再打印生成文本：

```text
cached/full max_abs_error=1.234e-07
```

如果 prompt 超过窗口长度，两条路径都会只读取末尾的 `max_seq_len` 个 token，误差检查也对这段实际可见的窗口进行。返回的文本仍保留完整 prompt，但模型看不到已经被截去的前半段。

Temperature、top-k 与 top-p 等采样参数也可以原样传入：

```bash
python exercises/block_03_transformer/task_30_kv_cache/kv_cache.py \
  --checkpoint /tmp/minimind_demo.pt --prompt "周一早晨，" \
  --max-new-tokens 20 --temperature 0.8 \
  --top-k 20 --top-p 0.9 --seed 0
```

参考：[Hugging Face：Caching](https://huggingface.co/docs/transformers/main/cache_explanation)、[Cache strategies](https://huggingface.co/docs/transformers/kv_cache)。
