# task_23：RoPE 位置编码

## 问题

RoPE 不把位置直接加到 embedding 上，而是旋转 Q/K，让注意力分数感知相对位置。

## 你要做什么

- 实现 RoPE 频率缓存。
- 实现 `apply_rope(q, k)`。

## 验收

- Q/K shape 不变。
- 成对维度被旋转。
- 能接入后续多头注意力。

