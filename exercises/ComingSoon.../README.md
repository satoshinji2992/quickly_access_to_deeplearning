# 拓展主题

这里放主线之外的深度学习主题.

主线课程先把三个问题讲清楚:

```text
基础训练闭环 -> ResNet 图像分类 -> MiniMind 文本生成
```

下面这些主题会在你继续往后读论文、看开源项目、训练小模型时反复出现. 它们不是杂项, 更像是主线之后自然会遇到的分叉路.

## Transformer 相关

- [Cross-Attention](./CrossAttention/README.md): 一个序列读取另一个序列, 常见于 encoder-decoder、多模态和扩散模型.
- [KV Cache](./KVCache/README.md): 自回归推理为什么可以加速.
- [MQA / GQA](./MQA_GQA/README.md): 为什么大模型会减少 KV heads.
- [Sampling](./Sampling/README.md): temperature、top-k、top-p 怎么影响生成.
- [Tokenizer](./Tokenizer/README.md): BPE、SentencePiece、词表大小为什么重要.

## 训练与对齐

- [SFT](./SFT/README.md): 指令微调怎么让模型更像在回答问题.
- [DPO / RLHF](./Alignment_DPO_RLHF/README.md): 偏好对齐的入口.
- [RL](./RL/README.md): PPO、GRPO 这类方法为什么会出现在大模型训练里.

## 架构拓展

- [MoE](./MoE/README.md): 专家混合, 为什么参数多但计算不一定等比例变大.
- [Mamba / State Space Model](./StateSpaceModels/README.md): Transformer 之外的序列建模路线.
- [Mamba](./Mamba/README.md): Mamba 方向的单独笔记.
- [GatedDeltaNet](./GatedDelataNet/README.md): 另一类序列模型尝试.
- [mHC](./mHC/README.md): 预留给后续架构笔记.
