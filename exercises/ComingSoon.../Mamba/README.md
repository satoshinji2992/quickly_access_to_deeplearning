# Mamba

Mamba 是 State Space Model 路线里很有代表性的模型.

它试图解决 Transformer 在长序列上 attention 成本高的问题.

## 关键词

- selective scan
- state space model
- 线性复杂度序列建模
- 长上下文

## 先怎么理解?

Transformer 像是在每一层都让 token 开会.

Mamba 更像维护一个会更新的状态, 让序列从左到右流过去.

这只是直觉, 但足够说明两者思路不同.

真正理解 Mamba 需要补 SSM、离散化、scan 并行化这些内容.
