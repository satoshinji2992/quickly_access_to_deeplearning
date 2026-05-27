#!/bin/zsh
set -e
content_dir='.blog_export/content/neural-networks-from-scratch'

transform() {
  local src="$1"
  local dest="$2"
  local title="$3"
  cat > "$content_dir/$dest" <<EOT
---
title: "$title"
date: 2026-04-13
draft: false
summary: "从零造神经网络专题教程"
---

EOT
  perl -0pe '
    s#\.\./assets/images/#/projects/from-scratch-nn/assets/images/#g;
    s#\(/Users/shenqi/Documents/code/深度学习入门/lessons/02-任务一实践引导\.md\)#(/neural-networks-from-scratch/07-practice-linear-regression/)#g;
    s#\(/Users/shenqi/Documents/code/深度学习入门/lessons/03-任务二实践引导\.md\)#(/neural-networks-from-scratch/08-practice-circle-classifier/)#g;
    s#\(/Users/shenqi/Documents/code/深度学习入门/lessons/02-从直线到梯度下降\.md\)#(/neural-networks-from-scratch/02-line-to-gradient-descent/)#g;
    s#\(/Users/shenqi/Documents/code/深度学习入门/lessons/03-圆形分类与MLP\.md\)#(/neural-networks-from-scratch/03-circle-classification-and-mlp/)#g;
    s#\(/Users/shenqi/Documents/code/深度学习入门/lessons/04-完善你的小型深度学习库\.md\)#(/neural-networks-from-scratch/04-build-your-mini-dl-lib/)#g;
    s#\(/Users/shenqi/Documents/code/深度学习入门/lessons/05-ResNet图像分类\.md\)#(/neural-networks-from-scratch/05-resnet-image-classification/)#g;
    s#\(/Users/shenqi/Documents/code/深度学习入门/lessons/06-Transformer与文本生成\.md\)#(/neural-networks-from-scratch/06-transformer-and-text-generation/)#g;
    s#\(/Users/shenqi/Documents/code/深度学习入门/exercises/task_00_linear_regression/mission_0\.py\)#(/projects/from-scratch-nn/repo/exercises/task_00_linear_regression/mission_0.py)#g;
    s#\(/Users/shenqi/Documents/code/深度学习入门/exercises/task_01_circle_classifier/mission_1\.py\)#(/projects/from-scratch-nn/repo/exercises/task_01_circle_classifier/mission_1.py)#g;
    s#\(/Users/shenqi/Documents/code/深度学习入门/exercises/task_04_transformer_textgen/README\.md\)#(/projects/from-scratch-nn/repo/exercises/task_04_transformer_textgen/README.md)#g;
    s#\(/Users/shenqi/Documents/code/深度学习入门/exercises/task_04_transformer_textgen/gpt_tutorial\.py\)#(/projects/from-scratch-nn/repo/exercises/task_04_transformer_textgen/gpt_tutorial.py)#g;
    s#\./01-%E8%AF%BE%E7%A8%8B%E4%B8%BB%E7%BA%BF\.md#(/neural-networks-from-scratch/01-course-map/)#g;
  ' "$src" >> "$content_dir/$dest"
}

cat > "$content_dir/_index.md" <<'EOT'
---
title: "从零造神经网络"
date: 2026-04-13
draft: false
summary: "从一条直线开始，一路走到 MLP、ResNet 和 Transformer 的深度学习入门专题。"
---

# 从零造神经网络

这是我把一套深度学习入门教程整理进博客后的专题页。

这套内容的主线一直很简单：

一条直线 -> 非线性分类 -> 手搓 MLP -> 小型深度学习库 -> ResNet -> Transformer

## 先看哪里

- [课程导读](/neural-networks-from-scratch/01-course-map/)
- [从直线到梯度下降](/neural-networks-from-scratch/02-line-to-gradient-descent/)
- [圆形分类与MLP](/neural-networks-from-scratch/03-circle-classification-and-mlp/)
- [完善你的小型深度学习库](/neural-networks-from-scratch/04-build-your-mini-dl-lib/)
- [ResNet图像分类](/neural-networks-from-scratch/05-resnet-image-classification/)
- [Transformer与文本生成](/neural-networks-from-scratch/06-transformer-and-text-generation/)

## 配套内容

- [任务一实践引导](/neural-networks-from-scratch/07-practice-linear-regression/)
- [任务二实践引导](/neural-networks-from-scratch/08-practice-circle-classifier/)
- [总演示动画](/projects/from-scratch-nn/docs/AI_Animation.html)
- [交互式教程](/projects/from-scratch-nn/docs/interactive.html)
EOT

cat > "$content_dir/01-course-map.md" <<'EOT'
---
title: "课程导读"
date: 2026-04-13
draft: false
summary: "从零造神经网络的整体学习路径。"
---

# 课程导读

这套教程现在已经拆成真正的章节，不再是一篇很长很长的主线文稿。

## 推荐顺序

1. [从直线到梯度下降](/neural-networks-from-scratch/02-line-to-gradient-descent/)
2. [圆形分类与MLP](/neural-networks-from-scratch/03-circle-classification-and-mlp/)
3. [完善你的小型深度学习库](/neural-networks-from-scratch/04-build-your-mini-dl-lib/)
4. [ResNet图像分类](/neural-networks-from-scratch/05-resnet-image-classification/)
5. [Transformer与文本生成](/neural-networks-from-scratch/06-transformer-and-text-generation/)

## 配套实践

- [任务一实践引导](/neural-networks-from-scratch/07-practice-linear-regression/)
- [任务二实践引导](/neural-networks-from-scratch/08-practice-circle-classifier/)

## 演示页

- [总演示动画](/projects/from-scratch-nn/docs/AI_Animation.html)
- [交互式教程](/projects/from-scratch-nn/docs/interactive.html)
EOT

transform 'lessons/02-从直线到梯度下降.md' '02-line-to-gradient-descent.md' '从直线到梯度下降'
transform 'lessons/03-圆形分类与MLP.md' '03-circle-classification-and-mlp.md' '圆形分类与MLP'
transform 'lessons/04-完善你的小型深度学习库.md' '04-build-your-mini-dl-lib.md' '完善你的小型深度学习库'
transform 'lessons/05-ResNet图像分类.md' '05-resnet-image-classification.md' 'ResNet图像分类'
transform 'lessons/06-Transformer与文本生成.md' '06-transformer-and-text-generation.md' 'Transformer与文本生成'
transform 'lessons/02-任务一实践引导.md' '07-practice-linear-regression.md' '任务一实践引导：拟合一条直线'
transform 'lessons/03-任务二实践引导.md' '08-practice-circle-classifier.md' '任务二实践引导：圆形分类与MLP'

cp -R assets/images/. .blog_export/static/projects/from-scratch-nn/assets/images/
cp docs/AI_Animation.html .blog_export/static/projects/from-scratch-nn/docs/AI_Animation.html
cp 'docs/深度学习入门_交互教程.html' .blog_export/static/projects/from-scratch-nn/docs/interactive.html
cp docs/index.html .blog_export/static/projects/from-scratch-nn/docs/index.html
mkdir -p .blog_export/static/projects/from-scratch-nn/repo
cp -R exercises .blog_export/static/projects/from-scratch-nn/repo/
cp -R lessons .blog_export/static/projects/from-scratch-nn/repo/
cp -R solutions .blog_export/static/projects/from-scratch-nn/repo/
cp README.md .blog_export/static/projects/from-scratch-nn/repo/
