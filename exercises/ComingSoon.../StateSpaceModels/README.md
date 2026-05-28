# Mamba / State Space Model

Transformer 用 attention 让 token 互相看.

代价是, 序列越长, attention 矩阵越大.

State Space Model 走的是另一条路: 用状态递推处理序列.

## SSM 的粗略直觉

你可以把它想成:

```text
当前输入 + 历史状态 -> 新状态 -> 输出
```

它不像 attention 那样显式构造所有 token 两两关系.

这让它在长序列上有另一种效率优势.

## Mamba

Mamba 是比较有代表性的 SSM 路线模型.

它引入 selective scan, 让模型根据输入动态调整状态更新.

粗略地说, 它想保留 RNN/SSM 的线性复杂度优势, 又提升表达能力.

## 和 Transformer 的关系

Mamba 不是“Transformer 已经过时”的简单替代.

更现实的看法是: 它提供了另一种序列建模工具.

长上下文、时间序列、需要高吞吐的场景里, SSM 很值得关注.
