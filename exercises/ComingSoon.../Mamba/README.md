# Mamba

Mamba 把 SSM 的部分参数设为当前输入的函数，使模型能够按 token 选择地保留或抑制信息。这一改动破坏了固定卷积核的直接形式，因此论文另外设计了 hardware-aware selective scan。

阅读时可把问题拆成三层：

1. SSM 的状态更新和离散化。
2. 输入依赖的选择参数如何改变信息保留。
3. selective scan 如何在硬件上执行该递推。

Mamba 与 causal attention 都可用于自回归序列建模，但二者的状态、并行方式和缓存语义不同。不应只用复杂度口号判断实际效果。

详见 [Mamba 原论文](https://arxiv.org/abs/2312.00752)。本目录尚未包含可运行实现。
