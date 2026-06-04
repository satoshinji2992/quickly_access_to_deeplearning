# 快速入门深度学习

你可能已经上过人工智能导论,也可能已经把作业文档丢给 AI 生成过一份看起来很完美的代码. 但代码能跑和你真的理解它,中间差得还挺远.

这个教程会带你从拟合一条直线开始,逐步搭出一个小型深度学习库,再到 ResNet 图像分类和 Transformer.Block2和Block3的内容有待精修,有愿意帮忙的小伙伴可以联系我.

[一些推荐的教学视频](推荐教学视频.md)

## 目录

```text
chapters/     # 章节
exercises/    # 练习代码
solutions/    # 参考实现
assets/       # 共享图片
tools/        # 小工具，比如手动裁图器
old/          # 旧稿
```

## 课程路线

1. [课程总览](chapters/00-课程总览.md)
2. [Block 1: 基础知识](chapters/01-基础知识.md) — 直线拟合 → 圆形分类 → 小型 NumPy DL 库 → MNIST
3. [Block 2: ResNet 图像分类](chapters/02-ResNet图像分类.md) — 卷积、池化、残差块、训练
4. [Block 3: Transformer 与 MiniMind](chapters/03-Transformer与MiniMind.md) — Attention、位置编码、Decoder、生成

## 小工具

如果你要把一张大图手动裁成各个章节和任务里要用的小图，可以直接用这个工具：

```bash
python tools/image_cropper.py
```

也可以在启动时顺手打开一张图：

```bash
python tools/image_cropper.py path/to/big_image.png
```

裁好的图片可以一次保存到多个位置，任务页和共享图片会一起对上。

### Block 1: y = ax + b! 神经网络到底是什么?

- [task_00: 拟合一条直线](exercises/block_01_basics/task_00_linear_regression/README.md)
- [task_01: 圆形分类](exercises/block_01_basics/task_01_circle_classifier/README.md)
- [task_02: 小型深度学习库](exercises/block_01_basics/task_02_mini_dl_lib/README.md)
- [task_03: MLP 识别 MNIST](exercises/block_01_basics/task_03_mnist_mlp/README.md)

### Block 2: 这是飞机还是轮船? 用 ResNet 分类物体!

- [task_10: 图像数据管线](exercises/block_02_resnet/task_10_image_data_pipeline/README.md)
- [task_11: Conv2D 与 im2col](exercises/block_02_resnet/task_11_conv2d_im2col/README.md)
- [task_12: 池化与 BatchNorm](exercises/block_02_resnet/task_12_pooling_and_bn/README.md)
- [task_13: 残差块](exercises/block_02_resnet/task_13_residual_block/README.md)
- [task_14: NumPy ResNet 训练](exercises/block_02_resnet/task_14_numpy_resnet_train/README.md)
- [task_15: 实验记录](exercises/block_02_resnet/task_15_experiment_notes/README.md)

### Block 3: apple is __ ? __ 注意力是你所需要的

- [task_20: Transformer 理论](exercises/block_03_transformer/task_20_transformer_theory/README.md)
- [task_21: Sinusoidal 位置编码](exercises/block_03_transformer/task_21_sinusoidal_position/README.md)
- [task_22: RoPE 位置编码](exercises/block_03_transformer/task_22_rope_position/README.md)
- [task_23: Causal Attention 与 GQA](exercises/block_03_transformer/task_23_causal_attention/README.md)
- [task_24: SwiGLU FFN](exercises/block_03_transformer/task_24_swiglu_ffn/README.md)
- [task_25: Embedding 与 LM Head](exercises/block_03_transformer/task_25_embedding_lm_head/README.md)
- [task_26: Decoder Block](exercises/block_03_transformer/task_26_decoder_blocks/README.md)
- [task_27: MiniMind Core](exercises/block_03_transformer/task_27_minimind_core/README.md)
- [task_28: Next-token 训练](exercises/block_03_transformer/task_28_next_token_training/README.md)
- [task_29: Generate 与采样](exercises/block_03_transformer/task_29_generate_sampling/README.md)
- [task_30: KV Cache](exercises/block_03_transformer/task_30_kv_cache/README.md)

## 拓展主题

`exercises/ComingSoon.../`

```text
CrossAttention
KVCache
MQA_GQA
Sampling
Tokenizer
SFT
Alignment_DPO_RLHF
MoE
Mamba / StateSpaceModels
RL
```

教程随时可能更新,使用这个命令同步:

```bash
git pull origin main
```
