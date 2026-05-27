---
title: "Transformer与文本生成"
date: 2026-04-13
draft: false
summary: "从零造神经网络专题教程"
---

# Transformer 与文本生成

这一部分目前还没有完全写成和前几章一样细的正文，但主方向已经确定了。

## 为什么会从 ResNet 走到 Transformer

前面图像部分处理的是二维空间结构，卷积在这种任务里很自然。

但文本不一样。文本是一个序列，模型不仅要看一个词，还要看它前后和更远位置上的关系。

所以这里要进入的新问题是：

- 模型怎么表示词
- 模型怎么知道顺序
- 模型怎么处理长距离依赖
- 模型怎么一步步生成下一个 token

## 这一部分最终会讲什么

后续会逐步补成下面这条线：

1. 为什么 RNN 有局限
2. 注意力机制在解决什么
3. Self-Attention 的基本计算
4. 多头注意力和前馈层
5. Transformer block
6. 文本生成训练流程

## 现在可以先看的项目

仓库里已经有一个可以直接运行的文本生成练习：

- [task_04_transformer_textgen](/Users/shenqi/Documents/code/深度学习入门/exercises/task_04_transformer_textgen)

建议先看：

- [README.md](/projects/from-scratch-nn/repo/exercises/task_04_transformer_textgen/README.md)
- [gpt_tutorial.py](/projects/from-scratch-nn/repo/exercises/task_04_transformer_textgen/gpt_tutorial.py)

## 这一部分后面怎么补

之后我会继续把它写成和前几章同样风格的正文，不直接从论文名词堆开始，而是尽量沿着：

`为什么需要它 -> 它在算什么 -> 最小代码长什么样 -> 项目怎么跑`

这条线展开。
