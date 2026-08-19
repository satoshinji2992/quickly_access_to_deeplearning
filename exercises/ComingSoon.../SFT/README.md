# SFT

SFT（Supervised Fine-Tuning）用带目标输出的样本继续训练预训练语言模型。损失仍是 next-token cross-entropy，改变的是数据分布和哪些 token 参与损失。

```text
instruction: 请解释什么是梯度下降
response:    梯度下降是一种优化方法……
```

对话模型通常先用 chat template 把多轮消息序列化，再构造 shifted labels。许多 SFT 配置只计算 assistant 内容的损失，将 system、user 和 padding 位置设为 `ignore_index`；这是训练选择，不是 next-token 目标的强制规定。

进行 SFT 实验时，至少要固定 tokenizer 和 chat template，统计有效 label 比例，并分开训练与验证对话。本页只是延伸笔记，尚未提供这套数据闭环。
