# task_22：正弦位置编码

## 问题

Attention 本身不知道顺序。位置编码把“第几个 token”写进向量。

## 你要做什么

- 实现 sin/cos 位置编码。
- 可视化不同维度上的位置变化。

## 验收

- 输出 shape 为 `(max_len, dim)`。
- 偶数维使用 sin，奇数维使用 cos。

