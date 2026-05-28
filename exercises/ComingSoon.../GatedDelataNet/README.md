# GatedDeltaNet

GatedDeltaNet 属于 Transformer 之外的序列建模尝试.

它和 Mamba、线性 attention、状态空间模型一样, 都在探索一个问题:

```text
能不能不用完整 attention 矩阵, 也建模长序列?
```

这类模型通常会关注:

- 长上下文效率.
- 训练稳定性.
- 推理吞吐.
- 和 Transformer 模块的混合使用.

这里先保留方向笔记.
