# 快速入门深度学习

这个教程的目的很简单: 带你把深度学习重新学一遍.

你可能已经上过人工智能导论, 也可能已经把作业文档丢给 AI 生成过一份看起来很完美的代码. 但代码能跑和你真的理解它, 中间差得还挺远.

所以这里不从一堆名词开始.

我们从一个具体问题开始: 怎么让电脑从数据里学出规律?

先拟合一条直线, 再判断点在圆内还是圆外, 然后把这些东西整理成一个小型深度学习库. 后面再进入图片分类和 Transformer.

这不是一本完整教材. 它更像一条练习路线. 你会读一点解释, 写一点代码, 跑一点实验, 然后遇到一些很烦但很有用的 bug.

---

## 从哪里开始?

建议按这个顺序走:

1. [课程总览](lessons/00-课程总览.md)
2. [Block 1: 基础知识](lessons/01-基础知识.md)
3. [Block 2: ResNet 图像分类](lessons/02-ResNet图像分类.md)
4. [Block 3: Transformer 与 MiniMind](lessons/03-Transformer与MiniMind.md)

如果你是第一次接触深度学习, 先做 Block 1.

不要一上来跳到 ResNet 或 Transformer. 后面的内容都会默认你已经知道训练循环、loss、梯度、反向传播和参数更新在做什么.

---

## 目录长什么样?

```text
.
├── lessons/      # 每一块的主线讲义
├── exercises/    # 每个小任务的说明和 starter 代码
├── solutions/    # 少量参考实现, 建议自己写完再看
├── assets/       # 教材图片
├── old/          # 旧稿, 留着当历史参考
└── README.md     # 当前入口
```

现在学生主要看 `lessons/` 和 `exercises/`.

`old/` 里是早期长文稿, 不删, 但不再作为主入口. 如果你想看这个教程最早的语气和写法, 可以翻它.

---

## Block 1: y = ax + b! 神经网络到底是什么?

第一块先别急着谈大模型.

我们从最小的训练闭环开始:

```text
forward -> loss -> backward -> update
```

你会从一条直线开始, 慢慢走到一个能训练的小型 NumPy 深度学习库.

任务入口:

- [task_00: 拟合一条直线](exercises/block_01_basics/task_00_linear_regression/README.md)
- [task_01: 使用神经网络预测点在圆形内还是圆形外](exercises/block_01_basics/task_01_circle_classifier/README.md)
- [task_02: 完善你的小型深度学习库](exercises/block_01_basics/task_02_mini_dl_lib/README.md)
- [task_03: 用 MLP 识别 MNIST](exercises/block_01_basics/task_03_mnist_mlp/README.md)

这一块做完以后, 你应该能说清楚参数、loss、梯度、学习率、激活函数、batch、epoch、优化器这些词到底在代码里对应什么.

---

## Block 2: 这是飞机还是轮船? 用 ResNet 分类物体!

第二块进入图像.

MNIST 可以 flatten 后交给 MLP, 但真实图片没那么好糊弄. 图片有局部结构, 有边缘, 有纹理, 有位置关系. 把它直接摊平成一串数字, 模型会很吃亏.

所以这里开始写 CNN 和 ResNet.

任务入口:

- [task_10: 图像数据管线](exercises/block_02_resnet/task_10_image_data_pipeline/README.md)
- [task_11: Conv2D 与 im2col](exercises/block_02_resnet/task_11_conv2d_im2col/README.md)
- [task_12: 池化与 BatchNorm](exercises/block_02_resnet/task_12_pooling_and_bn/README.md)
- [task_13: 残差块](exercises/block_02_resnet/task_13_residual_block/README.md)
- [task_14: NumPy ResNet 训练](exercises/block_02_resnet/task_14_numpy_resnet_train/README.md)
- [task_15: ResNet 实验记录](exercises/block_02_resnet/task_15_experiment_notes/README.md)

这一块会比 Block 1 硬很多. `Conv2D.backward`、`BatchNorm2D.backward`、残差块 shape 对齐, 都很容易写错.

别怕. 这块真正要练的是调试习惯: 看 shape, 做小输入测试, 做数值梯度检查, 先小样本过拟合.

---

## Block 3: apple is __ ? __ 注意力是你所需要的

第三块进入序列和语言模型.

图片模型处理的是空间结构, Transformer 处理的是 token 之间的关系.

这一块先讲 Attention Is All You Need 的基本发明, 再收束到 MiniMind 实现. Cross-attention、Tokenizer、SFT、DPO/RLHF、MoE、Mamba 这些放在拓展主题里.

任务入口:

- [task_20: Attention Is All You Need 到 MiniMind](exercises/block_03_transformer/task_20_transformer_theory/README.md)
- [task_21: Sinusoidal Position Encoding](exercises/block_03_transformer/task_21_sinusoidal_position/README.md)
- [task_22: RoPE 位置编码](exercises/block_03_transformer/task_22_rope_position/README.md)
- [task_23: Causal Attention 与 GQA](exercises/block_03_transformer/task_23_causal_attention/README.md)
- [task_24: SwiGLU 前馈网络](exercises/block_03_transformer/task_24_swiglu_ffn/README.md)
- [task_25: Embedding 与 LM Head](exercises/block_03_transformer/task_25_embedding_lm_head/README.md)
- [task_26: Decoder Block 叠加](exercises/block_03_transformer/task_26_decoder_blocks/README.md)
- [task_27: MiniMind Core](exercises/block_03_transformer/task_27_minimind_core/README.md)
- [task_28: Next-token 训练](exercises/block_03_transformer/task_28_next_token_training/README.md)
- [task_29: Generate 与采样](exercises/block_03_transformer/task_29_generate_sampling/README.md)
- [task_30: KV Cache](exercises/block_03_transformer/task_30_kv_cache/README.md)

这一块会用 PyTorch. 但注意, 不是为了让你直接调用现成模块糊过去. 该自己写的 RoPE、causal attention、SwiGLU、RMSNorm、embedding, 还是要自己写.

---

## 拓展主题

`exercises/ComingSoon.../` 里放主线之外的主题:

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

它们不是前三块的必经路线, 但都是读现代模型代码和论文时很容易遇到的东西.

---

## 怎么学比较舒服?

先读对应的 `lessons/`, 再进 `exercises/`.

不要一边看答案一边写. 你可以问 AI, 可以查文档, 可以让 AI 帮你解释报错, 但最好别直接让它替你把整关写完. 那样很快, 也很空.

遇到 backward 任务时, 先用很小的输入测试 shape.

比如:

```text
batch = 2
channels = 3
height = 4
width = 4
```

小输入看得见, 大输入只会让你更晕.

训练模型前, 先做小样本过拟合. 如果几十张、几百张样本都学不住, 不要急着跑完整数据集. 先查数据、shape、梯度、学习率.

实验记录不用写成报告. 记几句就够:

```text
我改了什么?
loss 怎么变?
accuracy 怎么变?
下次先查哪里?
```

---

## 运行环境

建议使用 macOS、Linux 或 WSL.

常用依赖大概是:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
```

部分任务会尝试使用 `torchvision` 下载 MNIST. 如果环境里没有 `torchvision`, 对应 starter 会退回到 `sklearn` 自带的小数据集, 至少能先把训练流程跑通.

---

## 同步代码

每次开始前可以同步:

```bash
git pull origin main
```

如果你已经改了本地练习代码, 先提交或备份自己的改动再同步.

那么, 开始吧.
