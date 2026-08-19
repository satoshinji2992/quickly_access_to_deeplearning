# 拓展主题

这里收录主线之外的延伸主题。多数条目目前是短笔记，不含 Block 1–3 那样的完整代码和测试。

建议先完成主线的三个项目：

```text
基础训练闭环 -> ResNet 图像分类 -> MiniMind 文本生成
```

再按需要选读以下笔记。

## Transformer 相关

- [Cross-Attention](./CrossAttention/README.md)：一个序列如何读取另一个序列。
- [KV Cache](./KVCache/README.md)：自回归推理中哪些中间结果可以复用。
- [MQA / GQA](./MQA_GQA/README.md)：减少 KV heads 如何降低 cache 开销。
- [Sampling](./Sampling/README.md)：temperature、top-k 和 top-p 如何改变候选分布。
- [Tokenizer](./Tokenizer/README.md)：BPE、SentencePiece 与词表大小的取舍。

## 训练与对齐

- [SFT](./SFT/README.md)：指令数据如何转换为 next-token 训练样本。
- [DPO / RLHF](./RL/Alignment_DPO_RLHF/README.md)：从成对偏好数据到对齐目标。
- [RL](./RL/README.md)：PPO 和 GRPO 在大模型后训练中的位置。

## 架构拓展

- [MoE](./MoE/README.md)：稀疏路由如何提高参数容量。
- [Mamba / State Space Model](./StateSpaceModels/README.md)：基于状态递推的序列建模路线。
- [Mamba](./Mamba/README.md)：selective SSM 的入门笔记。
- [GatedDeltaNet](./GatedDeltaNet/README.md)：带门控 delta rule 的线性序列模型。
- [mHC](./mHC/README.md)：流形约束的 Hyper-Connections 概念入口。
