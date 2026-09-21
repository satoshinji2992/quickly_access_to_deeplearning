# 自回归生成与采样

[Next-token 训练](../task_28_next_token_training/README.md) 能在一次 forward 中同时计算所有位置的 logits，因为真实前文已经出现在数据中。生成时的未来 token 尚不存在，每一轮都必须等上一轮做出选择，才能组成新的输入：

```text
当前前缀 -> 最后一位 logits -> 选 token -> 追加 -> 下一轮
```

未缓存版本会在每轮重新计算整个可见前缀。它的计算有重复，但流程直接，也能作为 [KV Cache](../task_30_kv_cache/README.md) 结果的数值参考。

![Greedy、temperature、top-k 与 top-p](assets/sampling_methods.png)

<div class="widget-mount" data-widget="sampling"></div>

## 为什么只取 `logits[:, -1]`

```text
input_ids: (B,T)
logits:    (B,T,V)
```

第 `t` 个位置的 logits 预测该位置之后的 token。当前前缀停在 `T-1`，因而只有最后一行分数对应尚未生成的下一个 token：

```python
next_logits = logits[:, -1]   # (B,V)
```

`sample_next_token` 为每个 batch 行返回一个 id，shape 为 `(B,1)`，生成循环再把它拼到 `input_ids` 末尾。若误取 `logits[:,0]`，每一轮都会根据序列开头那个位置选 token，前缀即使增长也没有被正确使用。

## 先跑一个最小生成循环

把上一页保存的 checkpoint 交给生成脚本，并使用 `temperature=0`，就能先避开随机采样的细节：

```bash
python exercises/block_03_transformer/task_29_generate_sampling/generate.py \
  --checkpoint /tmp/minimind_demo.pt --prompt "周一早晨，" \
  --max-new-tokens 20 --temperature 0
```

脚本内部的主循环可以先缩成四行看：

```python
for _ in range(max_new_tokens):
    logits, _ = model(input_ids)
    next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
    input_ids = torch.cat((input_ids, next_id), dim=1)
```

实际函数还要处理可见窗口、EOS、batch 和 padding mask，但不会改变这个基本时序。

## Greedy 直接取最大分数

这份 API 把 `temperature=0` 定义为 greedy decoding，直接取得分最高的 token：

```python
next_id = logits.argmax(dim=-1, keepdim=True)
```

数学上的 temperature 不能取 0，这里的 0 只是代码切换到 argmax 分支的约定，进入该分支后也不再应用 top-k 或 top-p。Greedy 的结果固定，很适合比较缓存前后的 token ids，只是局部最高分候选不一定会组成最自然的整段文本。

## Temperature 改变分布的尖锐程度

要从多个候选中随机抽取，需要先把 logits 变成概率。当 $\tau>0$ 时，设 $z$ 为最后位置的 logits 向量，temperature 缩放后的分布为

$$
p_i=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}.
$$

- `0 < τ < 1` 时，分布更尖，高 logit 占据更多概率。
- `τ > 1` 时，分布更平，低 logit 更容易被抽到。
- `τ < 0` 不在接口取值范围内。

所以 temperature 只改变如何从同一组 logits 抽样，不会修改模型参数，也不会重新计算 logits。

## Top-k 限制候选个数

Top-k 不改变保留项的相对大小，它找到第 `k` 大 logit 的阈值，将更低的候选设为负无穷，然后再做 Softmax：

```text
top_k=None  不过滤
top_k<=0    报错
top_k>V     按 V 处理
```

阈值处若有并列 logit，当前这种“按阈值”的实现会保留所有并列项，所以实际候选数可能超过 `k`。当最高 logit 唯一时，`top_k=1` 则与 greedy 产生同一 token。

## Top-p 按累计概率决定候选数

Top-p，也叫 nucleus sampling，不预先规定候选个数。它先按概率从高到低排序，再保留累计概率达到阈值所需的最小前缀。例如：

```text
probabilities = [0.60, 0.25, 0.10, 0.05]
top_p = 0.70
```

第一个候选累计概率只有 0.60，还没达到 0.70，因而要保留第二个，最终从前两项中抽样。也就是说，那个让累计概率首次跨过阈值的 token 也必须被保留。

```text
top_p=None 或 1  不过滤
0 < top_p < 1     nucleus 过滤
top_p<=0 或 >1    报错
```

两个过滤条件可以同时使用：代码先做 top-k，再在剩余候选中做 top-p。

## 随机数、EOS 和 batch

采样还涉及可复现性。CLI 为采样器创建独立的随机数生成器：

```python
generator = torch.Generator(device=device).manual_seed(seed)
```

在设备、软件环境、模型、prompt 和采样参数都相同时，同一 seed 会产生同一条采样序列；独立 generator 还能避免其他代码提前消耗全局随机数后改变结果。

`max_new_tokens` 限制新增 token 数。某个 batch 行先产生 EOS 后，后续轮次会继续为它填 EOS，其他行仍可继续，直到全部行结束时才提前停止。

解码时，`CharacterTokenizer.decode(..., skip_special_tokens=True)` 会隐藏 BOS、EOS、PAD 和 UNK，所以最终字符串长度不一定等于 token id 数；词表外字符映射成 UNK 后也不会显示。

## 变长 prompt 使用左 padding

生成循环统一读取每行最后一个位置，变长 batch 因而要在左侧放 PAD，使各行末尾都是真实 token：

```text
[PAD, PAD, BOS, t0, t1]  # 最后一位是有效 token
[BOS, t0, t1, PAD, PAD]  # 最后一位是 PAD，本接口拒绝
```

`attention_mask` 与 ids 同 shape，并且最后一列必须全部有效。没有显式传入 mask 时，代码会根据 `pad_token_id` 构造。当结果长度超过 `max_seq_len`，后续 forward 只能保留最近的可见窗口：

```python
context = result[:, -model.config.max_seq_len:]
```

用于返回的完整 `result` 仍保留所有已生成 id，但滑出可见窗口的旧 token 已经不再参与后续预测。

## 改成随机采样

去掉最小示例中的 `--temperature 0`，再传入 top-k 和 top-p，同一个 checkpoint 就会进入随机采样分支。例如从“周一早晨，”开始生成 20 个 token：

```bash
python exercises/block_03_transformer/task_29_generate_sampling/generate.py \
  --checkpoint /tmp/minimind_demo.pt --prompt "周一早晨，" \
  --max-new-tokens 20 --temperature 0.8 \
  --top-k 20 --top-p 0.9 --seed 0
```

脚本会先打印 checkpoint step，再打印 prompt 和生成文本。默认语料与模型都很小，文本可能重复或不通顺。这个未缓存版本每步都会重算前缀；[KV Cache](../task_30_kv_cache/README.md) 保留旧位置的 K/V，采样规则仍与本页一致。

参考：[The Curious Case of Neural Text Degeneration（nucleus sampling）](https://arxiv.org/abs/1904.09751)。
