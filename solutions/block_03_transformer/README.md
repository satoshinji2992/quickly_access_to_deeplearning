# Block 3 参考实现

这里放 Transformer 与 MiniMind Core 的参考实现。

当前文件：

- [minimind_solution.py](./minimind_solution.py)

它包含：

- RMSNorm
- RoPE cache 与 `apply_rope`
- GQA 风格 causal self-attention
- SwiGLU
- decoder block
- MiniMind 风格语言模型
- 简单 `generate`

运行训练：

```bash
python solutions/block_03_transformer/minimind_solution.py
```

这个脚本只用随机 token 跑几步训练，展示 MiniMind 参考实现的基本训练流程。
