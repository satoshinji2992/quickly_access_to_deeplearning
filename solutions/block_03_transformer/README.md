# Block 3 参考实现

Block 3 的章节目录已经包含可运行主线。为避免两份 Transformer 在后续修改中逐渐不一致，`minimind_solution.py` 不再复制模型代码，而是复用：

```text
task_27_minimind_core/minimind_core.py
task_30_kv_cache/kv_cache.py
```

参考入口额外提供 `MiniMindModel`。它继承 `MiniMindCore`，只把 `generate` 默认连接到 `generate_with_kv_cache`。

可直接导入：

```text
MiniMindConfig
RMSNorm
build_rope_cache / apply_rope
CausalSelfAttention
SwiGLU
DecoderBlock
MiniMindCore
MiniMindModel
generate_with_kv_cache
cache_equivalence_error
```

模型主干包含 token embedding、Pre-RMSNorm、causal RoPE/GQA、SwiGLU、两条 residual、final RMSNorm、共享 LM head 和逐层 KV Cache。`n_heads`、`n_kv_heads` 会改变实际投影与 cache shape。

## Smoke test

```bash
python solutions/block_03_transformer/minimind_solution.py
```

输出形如：

```text
logits: (1, 5, 64)
cached/full max_abs_error: 1.234e-07
generated: [[...共 9 个 token...]]
```

Smoke test 的输入长度为 5，greedy 生成 4 个 token，且不传 EOS，所以最终长度为 9。浮点误差尾数依设备而异，CPU float32 下通常小于 `1e-6`。

## 训练并比较两条生成路径

Checkpoint 生成命令：

```bash
python exercises/block_03_transformer/task_28_next_token_training/train.py \
  --steps 80 --checkpoint /tmp/minimind_demo.pt
```

未缓存采样：

```bash
python exercises/block_03_transformer/task_29_generate_sampling/generate.py \
  --checkpoint /tmp/minimind_demo.pt --prompt "清晨，" \
  --max-new-tokens 20 --temperature 0.8 \
  --top-k 20 --top-p 0.9 --seed 0
```

缓存采样：

```bash
python exercises/block_03_transformer/task_30_kv_cache/kv_cache.py \
  --checkpoint /tmp/minimind_demo.pt --prompt "清晨，" \
  --max-new-tokens 20 --temperature 0.8 \
  --top-k 20 --top-p 0.9 --seed 0
```

两条路径使用相同采样规则。`temperature=0` 提供确定性的 greedy token ids 对照；logits 分布与缓存的数值等价性不能由一条随机采样序列代替。

## 回归测试

```bash
python -m unittest discover -s tests -p 'test_block3.py' -v
```

测试覆盖正弦位置编码、RoPE 频率、GQA shape、causal 与上下文敏感性、weight tying、PAD mask、attention 梯度、单 batch 过拟合、train/validation 隔离、checkpoint round-trip、采样边界，以及 cached/full logits 和滑动窗口生成等价。

这套参考实现用于检查微型训练与生成管线。默认语料、字符 tokenizer、模型大小和训练步数不支持通用语言能力评估。
