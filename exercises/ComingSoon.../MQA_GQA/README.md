# MQA / GQA

标准 multi-head attention 为每个 query head 保留一组 K/V。MQA 让所有 query heads 共享一组 K/V；GQA 则让一组 query heads 共享一组 K/V。

```text
MHA: n_heads = 32, n_kv_heads = 32
GQA: n_heads = 32, n_kv_heads = 8
MQA: n_heads = 32, n_kv_heads = 1
```

上面的 GQA 例子中，每 4 个 query heads 共享 1 组 K/V。共享不会减少 query heads 的数量：attention 仍产生 32 个 head outputs，然后 concat 并做输出投影。

KV cache 的主要存储量与 `n_kv_heads` 成正比，因此 MQA/GQA 能减少自回归推理时的显存占用和内存带宽需求。共享的粒度则会影响质量与效率的取舍。

可运行实现见 [Task 23](../../block_03_transformer/task_23_causal_attention/README.md)，结构与实验背景见 [GQA 论文](https://arxiv.org/abs/2305.13245)。
