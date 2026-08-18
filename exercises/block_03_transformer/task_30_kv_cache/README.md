# task_30：KV Cache

未缓存生成在每一步重算整个可见窗口。Prompt 长度为 5 时：

```text
第 1 步：forward 5 tokens
第 2 步：forward 6 tokens
第 3 步：forward 7 tokens
...
```

旧 token 在每层得到的 K/V 不会因为后来又生成一个 token 而改变，因此推理时可以把它们留下。下一步只计算新 token 的 Q/K/V，再让新 Q 读取“历史 K/V + 当前 K/V”。

![每层 KV Cache：prefill 后逐 token 追加](assets/kv_cache.png)

KV Cache 用于自回归推理，不用于本章的并行训练 forward。

---

## Cache 保存什么

每个 decoder layer 有自己的 K/V 投影，cache 因此按层保存：

```text
past_key_values = [
    (K_layer0, V_layer0),
    (K_layer1, V_layer1),
    ...
]
```

每个张量的 shape 是：

```text
(B,n_kv_heads,past_len,head_dim)
```

这里存的是尚未 `repeat_kv` 的 K/V。若模型有 8 个 query heads、2 个 KV heads，cache 只保留 2 份 K/V，计算 attention 时再按组展开到 8 份。这也是 GQA 减少缓存体积的直接来源。

缓存并没有把 attention 变成常数时间。单步新 query 仍要和所有可见 keys 计算 score：

```text
Q:       (B,n_heads,1,head_dim)
K_cache: (B,n_heads,past_len+1,head_dim)  # 计算时展开
scores:  (B,n_heads,1,past_len+1)
```

它省掉的是旧 token 的 block 计算和 K/V 投影；cache 本身的存储会随上下文长度增长。

## `prefill` 与 `decode_one`

`prefill` 一次处理整个可见 prompt：

```python
logits, cache = prefill(model, input_ids, attention_mask)
```

返回所有 prompt 位置的 logits 和每层 K/V。实际生成第一个 token 只用 `logits[:, -1]`，保留全部 logits 是为了后面的数值对照。

得到第一个新 token 后，`decode_one` 只接受 `(B,1)`：

```python
step_logits, cache = decode_one(
    model,
    next_id,             # (B,1)
    past_key_values,
    attention_mask,
)
```

每层把新 K/V 沿序列轴追加，长度从 `past_len` 变为 `past_len+1`。

传入 mask 时，其范围覆盖缓存前缀和当前 token：

```text
attention_mask: (B,past_len+1)
```

`decode_one` 在缺省 mask 时把整个前缀视为有效。Prompt 含左侧 PAD 时，完整 mask 会随新 token 保留并追加 `True`；如果使用缺省值，原有 PAD 位置会被视为有效。右 padding 仍会让最后一行 logits 落在 PAD query 上，因此生成接口会拒绝。

## RoPE 的位置接着 `past_len`

缓存已有 `past_len` 个位置时，新 token 的位置索引就是 `past_len`：

```python
cos, sin = build_rope_cache(
    seq_len=1,
    head_dim=head_dim,
    start_pos=past_len,
)
```

若每一步又从位置 0 开始，shape、cache 长度甚至生成循环都可能看似正常，但 Q/K 旋转角度已经偏离。Cached/full logits 的数值比较会直接暴露这种静默错误。

## 怎样做等价测试

`logits_with_kv_cache` 采用最容易检查的路径：先 prefill 第一个 token，之后逐 token 调用 `decode_one`，最后拼出所有位置的 cached logits。

```python
full_logits, _ = model(input_ids, attention_mask=mask)
cached_logits = logits_with_kv_cache(model, input_ids, mask)
error = (full_logits - cached_logits).abs().max()
```

CPU float32 下最大误差通常在 `1e-6` 以内；具体尾数取决于设备和浮点运算次序。Logits 使用浮点容差比较，greedy token ids 则逐项完全相同。

这个对照同时覆盖：

- 每层 cache 是否对应正确；
- K/V 是否沿序列轴追加；
- RoPE 的 `start_pos` 是否正确；
- 非方阵 causal mask 是否对齐当前 query；
- padding mask 是否覆盖历史前缀。

因此，仅观察“代码里有一个 list”或“生成能跑完”无法区分上述实现细节。

## 缓存生成循环

```text
prompt -> prefill -> last logits + cache

repeat:
    sample next_id
    append next_id
    decode_one(next_id, cache)
```

Temperature、top-k、top-p、EOS 和 batch 结束规则与 Task 29 相同。`temperature=0` 时可以直接比较：

```text
cached greedy ids == ordinary greedy ids
```

某个 batch 行先遇到 EOS 后会持续填 EOS，其余行继续，直到所有行结束或达到 `max_new_tokens`。

## 窗口满时为什么重新 prefill

Cache 长度不能超过 `max_seq_len`。窗口已满又生成新 token 时，参考实现会保留最近窗口，并把其中第一个 token 重新视作位置 0：

```python
visible = result[:, -max_seq_len:]
```

旧 cache 中的 RoPE 位置基于滚动前的索引，直接裁掉最早 K/V 并不能得到参考实现的新位置语义。因此缓存版本在窗口滚动时重新 prefill。那一步暂时失去增量计算优势，但能保证两条教学实现一致。

这是本仓库选择的有限窗口策略，不代表已经实现更复杂的长期位置外推或 ring-buffer cache。

## 运行与核对

Checkpoint 可由 Task 28 的训练脚本产生：

```bash
python exercises/block_03_transformer/task_28_next_token_training/train.py \
  --steps 80 --checkpoint /tmp/minimind_demo.pt
```

Greedy 缓存生成的命令是：

```bash
python exercises/block_03_transformer/task_30_kv_cache/kv_cache.py \
  --checkpoint /tmp/minimind_demo.pt --prompt "清晨，" \
  --max-new-tokens 20
```

第一行形如：

```text
cached/full max_abs_error=1.234e-07
```

CPU float32 下通常小于 `1e-6`。第二行是生成文本；默认模型很小，文本流畅度不用来判断 cache 是否正确。

也可以检查缓存采样参数：

```bash
python exercises/block_03_transformer/task_30_kv_cache/kv_cache.py \
  --checkpoint /tmp/minimind_demo.pt --prompt "清晨，" \
  --max-new-tokens 20 --temperature 0.8 \
  --top-k 20 --top-p 0.9 --seed 0
```

缓存性质也收录在 Block 3 测试中：

```bash
python -m unittest discover -s tests -p 'test_block3.py' -v
```

测试覆盖 cache 层数与 `n_kv_heads` shape、每次 decode 后的长度变化、cached/full logits 容差、两种 greedy 生成的 token 一致性、padding mask 沿 cache 的传递，以及跨过 `max_seq_len` 后与未缓存窗口策略的一致性。

参考：[Hugging Face：Caching](https://huggingface.co/docs/transformers/main/cache_explanation)、[Cache strategies](https://huggingface.co/docs/transformers/kv_cache)。
