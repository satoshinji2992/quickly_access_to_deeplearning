# Block 3 参考实现

文件：

- [minimind_solution.py](./minimind_solution.py)

`minimind_solution.py` 包含：

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

这个脚本使用随机 token 运行几步训练，展示 MiniMind 的基本训练流程。
