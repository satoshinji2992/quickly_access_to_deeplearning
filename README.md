# 快速入门深度学习

> 从 `y = ax + b` 出发，一路写到 `token → logits`。

[查看项目主页](https://deep-learning-field-notes.wwwsatoshinji.chatgpt.site) · [打开课程总览](chapters/00-课程总览.md) · [浏览论文与视频资料](推荐教学视频.md)

这里不从厚厚的 API 手册开始。第一站只是一条直线；接着拆开前向传播与反向传播，搭一套小型 NumPy 深度学习库，再让 CIFAR-100 图像流过 ResNet，最后拼出带 RoPE、GQA 和 KV Cache 的 decoder-only Transformer。

代码能跑只是起点。每个可运行主题都附有入口、命令和结果核对方法，用数据隔离、梯度、shape 与 checkpoint round-trip 判断计算是否真的对上了。

**三段路线，一条主线：**

- **Block 1 · 看清梯度** — 线性回归 → 圆形分类 → Mini DL Library → MNIST
- **Block 2 · 看清像素流** — CIFAR-100 → Conv2D → BatchNorm → ResNet
- **Block 3 · 看清上下文** — Attention → RoPE / GQA → MiniMind → KV Cache

## 环境准备

推荐使用 conda 管理 Python 环境，用 `requirements.txt` 安装项目依赖。该文件覆盖完整教程，包括 NumPy、MNIST 和 Transformer 相关依赖：

```bash
conda create -n dl_tutorial python=3.10
conda activate dl_tutorial
pip install -r requirements.txt
```

如果只学习 Block 1 和 Block 2 的 NumPy 部分，核心依赖是 `numpy`、`pandas`、`matplotlib` 和 `scikit-learn`。`torch` / `torchvision` 主要用于 MNIST 与 Transformer 相关任务。

## 目录

```text
chapters/     # 章节
exercises/    # 分节代码与可运行示例
solutions/    # 参考实现
common/       # task_02 之后共享的小型 NumPy 深度学习库完整实现
assets/       # 共享图片
```

关键配图中的矩阵数值、shape、梯度数量和 head 映射都有自动检查。task 15 的误分类图来自真实 CIFAR-100 推理，并保留 test split、checkpoint 摘要和测试索引。

## 从哪里开始

1. [课程总览](chapters/00-课程总览.md)
2. [Block 1: 基础知识](chapters/01-基础知识.md) — 直线拟合 → 圆形分类 → 小型 NumPy DL 库 → MNIST
3. [Block 2: ResNet 图像分类](chapters/02-ResNet图像分类.md) — 卷积、池化、残差块、训练
4. [Block 3: Transformer 与 MiniMind](chapters/03-Transformer与MiniMind.md) — Attention、位置编码、Decoder、生成

### Block 1: y = ax + b! 神经网络到底是什么?

- [task_00: 拟合一条直线](exercises/block_01_basics/task_00_linear_regression/README.md)
- [task_01: 圆形分类](exercises/block_01_basics/task_01_circle_classifier/README.md)
- [task_02: 小型深度学习库](exercises/block_01_basics/task_02_mini_dl_lib/README.md)
- [task_03: MLP 识别 MNIST](exercises/block_01_basics/task_03_mnist_mlp/README.md)

### Block 2: 用 ResNet 分类 CIFAR-100 小图像

- [task_10: 图像数据管线](exercises/block_02_resnet/task_10_image_data_pipeline/README.md)
- [task_11: Conv2D 与 im2col](exercises/block_02_resnet/task_11_conv2d_im2col/README.md)
- [task_12: 池化与 BatchNorm](exercises/block_02_resnet/task_12_pooling_and_bn/README.md)
- [task_13: 残差块](exercises/block_02_resnet/task_13_residual_block/README.md)
- [task_14: NumPy ResNet 训练](exercises/block_02_resnet/task_14_numpy_resnet_train/README.md)
- [task_15: 实验记录](exercises/block_02_resnet/task_15_experiment_notes/README.md)

### Block 3: 用上下文补出下一个 token

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

## 还想继续往前？

`exercises/ComingSoon.../`

```text
CrossAttention
KVCache
MQA_GQA
Sampling
Tokenizer
SFT
RL/Alignment_DPO_RLHF
MoE
Mamba / StateSpaceModels
RL
```

同步上游更新：

```bash
git pull origin main
```

运行参考实现：

```bash
python solutions/block_01_basics/linear_regression_solution.py
python solutions/block_01_basics/mini_network_reference.py
python solutions/block_02_resnet/train_cifar100_solution.py --subset-size 200 --epochs 20 --batch-size 20 --channels 8 16 32 --lr 0.03
python solutions/block_03_transformer/minimind_solution.py
```

不下载数据的快速检查：

```bash
python -m unittest discover -s tests -p 'test_block1.py' -v
python -m unittest discover -s tests -p 'test_block2.py' -v
python -m unittest discover -s tests -p 'test_block3.py' -v
python -m unittest tests.test_docs -v
```

预期每条命令末尾都显示 `OK`。Block 2/3 的测试覆盖 shape、梯度、数据隔离、causal 性质和 checkpoint 等基础约束；它们不衡量完整 CIFAR-100 训练精度，也不代表模型具备通用语言能力。
