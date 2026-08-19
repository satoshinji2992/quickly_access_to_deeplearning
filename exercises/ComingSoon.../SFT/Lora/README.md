# LoRA

LoRA 冻结预训练权重 `W`，用两个可训练的低秩矩阵表示增量：

$$
W' = W + \frac{\alpha}{r}BA,
$$

其中 `rank(A) = rank(B) = r` 的说法并不严格；更准确地说，`A` 和 `B` 的内部维度是 `r`，因而 `BA` 的秩不超过 `r`。

例如 `W ∈ R^{d_out×d_in}` 时，可取 `A ∈ R^{r×d_in}`、`B ∈ R^{d_out×r}`。当 `r` 远小于 `d_in` 和 `d_out` 时，可训练参数从 `d_out·d_in` 降为 `r(d_in+d_out)`。

LoRA 常用于 SFT，但“参数更少”不等于“总训练内存按同样比例下降”：激活、优化器状态、序列长度和目标层选择仍会影响实际开销。实验应记录 `r`、`alpha`、dropout、target modules 与是否已合并权重。

定义和实验背景见 [LoRA 原论文](https://arxiv.org/abs/2106.09685)。本目录尚未包含可运行实现。
