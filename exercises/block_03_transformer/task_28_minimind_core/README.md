# task_28：MiniMind Core

## 问题

把前面的模块整理成对齐 MiniMind 思路的小型 decoder-only 模型。

## 你要做什么

- 实现配置类。
- 实现 RMSNorm。
- 接入 RoPE、GQA、SwiGLU。
- 搭出 MiniMind 风格 decoder-only 模型。

## 验收

- forward 输出 `(batch, seq_len, vocab_size)`。
- 能打印参数量。
- 支持保存和加载 `state_dict`。

