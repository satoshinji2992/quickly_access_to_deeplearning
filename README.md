# 快速入门深度学习

这是一套面向初学者的实践型深度学习教程。它不从一堆名词开始，而是从一个具体问题开始：怎么让模型从数据里学会规律？

课程按三块推进。每一块都有主线讲义、任务说明、starter 代码和参考实现。你可以一边读，一边改代码，一边做实验。

## 从哪里开始

建议按这个顺序走：

1. [课程总览](lessons/00-课程总览.md)
2. [Block 1：基础知识](lessons/01-基础知识.md)
3. [Block 2：ResNet 图像分类](lessons/02-ResNet图像分类.md)
4. [Block 3：Transformer 与 MiniMind](lessons/03-Transformer与MiniMind.md)

如果你是第一次接触深度学习，先做 Block 1，不要直接跳到 ResNet 或 Transformer。后面的内容都会默认你已经知道训练循环、loss、梯度、反向传播和参数更新在做什么。

## 课程结构

```text
.
├── lessons/      # 每一块的主线讲义
├── exercises/    # 学生任务：README + starter 代码
├── solutions/    # 参考实现，建议完成任务后再看
├── assets/       # 教材图片资源
└── README.md     # 当前入口
```

旧版 `notes/` 和 `codes/` 已从当前主线中移除。它们只保留在 Git 历史里，不再作为学生阅读入口。现在以 `lessons/` 和 `exercises/` 为准。

## Block 1：基础知识

Block 1 解决的是：深度学习最小闭环到底是什么？

你会从线性回归开始，亲手写出：

- 模型参数
- 损失函数
- 梯度下降
- 标准化
- MLP
- ReLU / Softmax / Cross Entropy
- forward / backward
- 一个小型 NumPy 深度学习库

任务入口：

- [task_00：线性回归](exercises/block_01_basics/task_00_linear_regression/README.md)
- [task_01：圆形分类与 MLP](exercises/block_01_basics/task_01_circle_classifier/README.md)
- [task_02：完善小型深度学习库](exercises/block_01_basics/task_02_mini_dl_lib/README.md)
- [task_03：MNIST MLP 过渡](exercises/block_01_basics/task_03_mnist_mlp/README.md)

## Block 2：ResNet 图像分类

Block 2 解决的是：图片为什么不能只靠 flatten + MLP？

你会继续沿着 NumPy 实现路线，写出图像分类需要的核心模块：

- 图像数据管线
- `NCHW` 张量组织
- `Conv2D`
- `im2col` / `col2im`
- `MaxPool2D`
- `BatchNorm2D`
- `BasicBlock`
- 轻量 ResNet 训练流程
- 消融实验报告

任务入口：

- [task_10：图像数据管线](exercises/block_02_resnet/task_10_image_data_pipeline/README.md)
- [task_11：Conv2D 与 im2col](exercises/block_02_resnet/task_11_conv2d_im2col/README.md)
- [task_12：池化与 BatchNorm](exercises/block_02_resnet/task_12_pooling_and_bn/README.md)
- [task_13：残差块](exercises/block_02_resnet/task_13_residual_block/README.md)
- [task_14：NumPy ResNet 训练](exercises/block_02_resnet/task_14_numpy_resnet_train/README.md)
- [task_15：ResNet 实验报告](exercises/block_02_resnet/task_15_resnet_report/README.md)

## Block 3：Transformer 与 MiniMind

Block 3 解决的是：模型怎么处理 token 序列和上下文关系？

这一块会从 attention 的基本计算开始，逐步搭到一个小型语言模型训练任务：

- Transformer 理论
- 交叉注意力
- 余弦位置编码
- RoPE 位置编码
- 多头注意力
- 前馈网络
- embedding 和 LM head
- 多层 Transformer 叠加
- MiniMind 风格核心结构
- 具体训练任务

任务入口：

- [task_20：Transformer 理论](exercises/block_03_transformer/task_20_transformer_theory/README.md)
- [task_21：交叉注意力实现](exercises/block_03_transformer/task_21_cross_attention/README.md)
- [task_22：余弦位置编码](exercises/block_03_transformer/task_22_sinusoidal_position/README.md)
- [task_23：RoPE 位置编码](exercises/block_03_transformer/task_23_rope_position/README.md)
- [task_24：多头注意力](exercises/block_03_transformer/task_24_multi_head_attention/README.md)
- [task_25：前馈网络](exercises/block_03_transformer/task_25_feedforward/README.md)
- [task_26：Embedding 接入](exercises/block_03_transformer/task_26_embedding_and_lm_head/README.md)
- [task_27：多层叠加](exercises/block_03_transformer/task_27_stack_transformer_blocks/README.md)
- [task_28：MiniMind 核心结构](exercises/block_03_transformer/task_28_minimind_core/README.md)
- [task_29：具体训练任务](exercises/block_03_transformer/task_29_minimind_training/README.md)

## 怎么学习

每个任务都按同一节奏写：

1. 这个任务解决什么问题
2. 如何让模型学会这个？
3. 需要引入哪些数学对象
4. 对应代码该怎么组织
5. 逐步实现顺序
6. 常见错误和调试方法
7. 必做实验
8. 完成后应该能回答的问题
9. 下一任务为什么自然出现

建议你每做完一个任务，都用自己的话回答第 8 节的问题。能讲清楚，才是真的过关。

## 使用建议

- 先读 `lessons/`，再进对应 `exercises/`。
- 先自己写 starter，再看 `solutions/`。
- 每次改完一层，都用小输入做 shape 检查。
- 涉及 backward 的任务，要做数值梯度检查。
- 训练模型前，先做小样本过拟合测试。
- 写报告时，一次只改一个变量。

## 同步代码

每次开始前可以同步：

```bash
git pull origin main
```

如果你已经改了本地练习代码，先提交或备份自己的改动，再同步仓库。
