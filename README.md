# 深度学习入门

一个面向初学者的实践型深度学习教程，从 `y = ax + b` 出发，手搓前向传播、反向传播、损失函数、优化器、激活函数，再逐步走到图像分类中的 ResNet 和文本生成中的 Transformer。

## 从哪里开始

建议按下面顺序阅读和动手：

1. [docs/学习路线.md](/Users/shenqi/Documents/code/深度学习入门/docs/学习路线.md)
2. [lessons/01-课程主线.md](/Users/shenqi/Documents/code/深度学习入门/lessons/01-课程主线.md)
3. [lessons/02-从直线到梯度下降.md](/Users/shenqi/Documents/code/深度学习入门/lessons/02-从直线到梯度下降.md)
4. [lessons/03-圆形分类与MLP.md](/Users/shenqi/Documents/code/深度学习入门/lessons/03-圆形分类与MLP.md)
5. [lessons/04-完善你的小型深度学习库.md](/Users/shenqi/Documents/code/深度学习入门/lessons/04-完善你的小型深度学习库.md)
6. [lessons/05-ResNet图像分类.md](/Users/shenqi/Documents/code/深度学习入门/lessons/05-ResNet图像分类.md)
7. [lessons/06-Transformer与文本生成.md](/Users/shenqi/Documents/code/深度学习入门/lessons/06-Transformer与文本生成.md)

如果你想先快速浏览整个仓库的结构和定位，可以看：

- [docs/教程重构规划.md](/Users/shenqi/Documents/code/深度学习入门/docs/教程重构规划.md)
- [docs/深度学习入门_交互教程.html](/Users/shenqi/Documents/code/深度学习入门/docs/深度学习入门_交互教程.html)
- [docs/AI_Animation.html](/Users/shenqi/Documents/code/深度学习入门/docs/AI_Animation.html)

## GitHub Pages

仓库已经补好 GitHub Actions 部署配置：

- [deploy-pages.yml](/Users/shenqi/Documents/code/深度学习入门/.github/workflows/deploy-pages.yml)

推送到 `main` 后，GitHub Actions 会自动把 `docs/` 和 `assets/` 发布到 GitHub Pages。

建议发布入口：

- `https://<你的用户名>.github.io/<仓库名>/docs/`

默认会跳转到：

- `docs/AI_Animation.html`

## 当前目录

```text
.
├── README.md
├── docs/         # 路线图、重构说明、交互式 HTML 入口
├── lessons/      # 教程正文与实践引导
├── exercises/    # 当前可运行练习与项目
├── solutions/    # 参考实现
└── assets/       # 教程图片等静态资源
```

## 重点内容

- `lessons/01-课程主线.md`
  课程导读页，负责把整条学习路径串起来。

- `lessons/02-从直线到梯度下降.md`
  任务一主线正文，讲模型、损失函数、梯度下降和训练闭环。

- `lessons/03-圆形分类与MLP.md`
  任务二主线正文，讲非线性分类、激活函数、MLP、交叉熵和反向传播。

- `lessons/04-完善你的小型深度学习库.md`
  任务三主线正文，讲激活函数、优化器、归一化、正则化和初始化。

- `lessons/05-ResNet图像分类.md`
  图像分类主线，讲卷积、残差块和 ResNet 实验流程。

- `exercises/task_00_linear_regression/`
  任务一：最小线性回归练习。

- `exercises/task_01_circle_classifier/`
  任务二：用多层感知机判断点在圆内还是圆外。

- `exercises/task_04_transformer_textgen/`
  进阶项目：基于《红楼梦》文本的 Transformer 生成 demo。

- `solutions/`
  放的是参考实现，建议先自己做，再来对照看。

## 使用建议

- 先读主线，再做练习，不要一上来就直接看 `solutions/`
- 图像和文本部分建议使用 PyTorch
- 仓库已经切换到新结构，后续会继续在 `lessons/`、`exercises/` 和 `solutions/` 上迭代

## 更新

仓库会继续更新。每次开始前建议先同步：

```bash
git pull origin main
```
