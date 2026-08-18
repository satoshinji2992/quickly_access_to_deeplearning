# MoE

Mixture of Experts（MoE）通常用多个 expert FFN 替换密集 FFN。Router 为每个 token 计算 expert 分数，只激活 top-k 个 expert，再按路由权重合并输出。

```text
8 experts, top_k = 2
token -> router -> 2 selected experts -> weighted sum
```

这种稀疏激活可以在每个 token 计算量增长较慢的情况下增加总参数容量。不过，实际速度还受 token dispatch、all-to-all 通信、expert capacity 和硬件利用率影响，不能只由 top-k 推断。

常见失效模式是路由过度集中：少数 experts 过载，其余 experts 得不到训练。因此训练中往往需要 load-balancing 目标，并记录每个 expert 的 token 数和丢弃率。

本页尚未提供 router、dispatch 或分布式通信实现。
