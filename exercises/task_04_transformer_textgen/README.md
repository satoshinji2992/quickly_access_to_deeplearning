# 使用Transformer生成《红楼梦》文本

基于Transformer架构的中文古典文学生成模型，通过学习《红楼梦》的语言风格和叙事结构，能够生成风格一致的文本内容。

## 项目结构

- `gpt_tutorial.py`: 主要模型实现和训练代码
- `asset/Hong_Lou_Meng.txt`: 训练数据集
- [训练结果展示](./RESULT.md): 模型训练日志和生成样本

## 参考教程
[PyTorch手搓Transformer，生成《红楼梦》风格文本](https://space.bilibili.com/1570063857/lists/4675218?type=season/)

## 使用方法

### 环境要求
- Python 3.8+
- PyTorch 1.9+
- CUDA支持（可选，但推荐）

### 训练模型
```bash
python gpt_tutorial.py