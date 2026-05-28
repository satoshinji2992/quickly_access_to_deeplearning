# MoE

MoE 是 Mixture of Experts, 专家混合.

普通 FFN 每次都会使用全部参数.

MoE 不一样. 它准备很多个 expert, 每个 token 只选择其中几个 expert 计算.

## 为什么参数多但计算不等比例变大?

假设有 8 个 expert.

每个 token 只走 2 个:

```text
top_k = 2
```

那模型总参数很多, 但每个 token 实际激活的参数只是一部分.

这叫 sparse activation.

## Router

MoE 需要一个 router 决定 token 交给哪些 expert.

router 输出每个 expert 的分数, 选 top-k.

问题也在这里:

- 有些 expert 可能太忙.
- 有些 expert 可能几乎没人用.
- 训练时要加 load balancing loss.

## MoE 的直觉

不是每个 token 都需要同一套 FFN.

代码 token、数学 token、聊天 token, 可能适合不同 expert.

MoE 让模型容量变大, 但不让每次计算都变得同样大.
