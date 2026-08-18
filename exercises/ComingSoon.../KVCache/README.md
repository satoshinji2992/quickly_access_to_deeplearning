# KV Cache

自回归解码每次只追加一个 token。历史 token 在各层产生的 K/V 不会因新 token 而改变，因此可以缓存；当前步只计算新的 Q/K/V，再把新 K/V 追加到 cache。

```text
k_cache: (B, H_kv, T_past, D_h)
v_cache: (B, H_kv, T_past, D_h)
new K/V: (B, H_kv, 1, D_h)
```

每一层 attention 都需要独立的 K/V cache。缓存减少了历史 token 的重复投影和 attention 计算，但不改变模型权重或生成分布。

实现时需同时检查：

- RoPE 位置应从 `past_len` 开始，不能在每步归零。
- GQA 的 cache 按 `n_kv_heads` 存储，不是 `n_heads`。
- padding、分批解码和滑动窗口会改变有效 key 范围。
- 最直接的正确性检查是：同一输入下，cached 与 uncached 的每步 logits 应在浮点误差内一致。

主线中的可运行实现见 [Task 30](../../block_03_transformer/task_30_kv_cache/README.md)；工程语义可对照 [Hugging Face cache 文档](https://huggingface.co/docs/transformers/main/cache_explanation)。
