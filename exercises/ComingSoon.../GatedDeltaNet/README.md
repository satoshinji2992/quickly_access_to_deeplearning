# Gated DeltaNet

Gated DeltaNet 是一类线性时间序列模型。它把递推状态视为可在线更新的联想记忆：delta rule 根据当前 key/value 修正状态，衰减门则控制旧信息保留多少。

它与 Mamba、线性 attention 和其他递推模型的共同问题是：不显式构造完整 attention 矩阵时，如何保留内容相关的长程信息，并同时获得可并行的训练算法。

要把该主题纳入主线，还缺少状态更新公式的逐项推导、并行 scan 与递推版的等价测试、数值稳定性以及吞吐实验。当前页面不包含这些实现。

原始方法见 [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://proceedings.iclr.cc/paper_files/paper/2025/file/4904fad153f6434a7bcf04465d4be2cc-Paper-Conference.pdf)。
