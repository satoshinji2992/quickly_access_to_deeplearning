# Embedding 与 LM head

语言模型的中间部分处理向量，输入与输出却都要回到 token 这一层。Embedding 把 id 变成向量，LM head 再把上下文向量变成对整个词表的打分，两者分别位于模型的入口和出口：

```text
token ids (B,T)
    -> Embedding
hidden states (B,T,D)
    -> Transformer
contextual states (B,T,D)
    -> LM head
logits (B,T,V)
```

这条数据流中，`B` 是 batch 大小，`T` 是序列长度，`D` 是向量宽度，`V` 是词表大小。

## 从自己保存参数到 `nn.Module`

前两章的层自己保管 `W`、`b` 和对应梯度，组合模型再逐层收集参数。PyTorch 的 `nn.Module` 接手了这部分工作：在 `__init__` 中把子层赋给 `self`，模块就会登记它们；`model.parameters()` 能递归取到这些子层的可训练 Parameter，优化器不必再手动寻找每一张权重表。

`forward` 仍然由我们写，调用 `model(input_ids)` 时会进入这个方法。与 NumPy 不同，PyTorch 会记录可求导的运算，因此只需对标量 loss 调用一次 `backward()`。拿第一章的直线做一个很小的对照：

```python
import torch

a = torch.nn.Parameter(torch.tensor(1.0))
b = torch.nn.Parameter(torch.tensor(0.0))
prediction = a * 2.0 + b
loss = (prediction - 3.0).square()
loss.backward()
print(a.grad.item(), b.grad.item())  # -4.0 -2.0
```

因为当前预测为 2，误差为 -1，手算得到 $\partial L/\partial a=2(-1)\times2=-4$，$\partial L/\partial b=2(-1)=-2$，与自动求导一致。Embedding 的权重也是 Parameter；整数 id 只是索引，不需要也不能像连续权重一样求导。

![Embedding 表与共享权重的 LM head](assets/embedding_lm_head.png)

<div class="widget-mount" data-widget="token-embed-3d"></div>

## Embedding 就是按 id 查行

设词表中有 `V` 个 token，每个 token 用 `D` 个数表示，Embedding 权重 $E$ 的 shape 就是 `(V,D)`，其中每一行对应一个 token：

```text
E[0]  -> token 0 的向量
E[1]  -> token 1 的向量
...
```

当 `input_ids[b,t]` 等于 `v` 时，该位置取出 `E[v]`。id 没有大小意义，只决定查哪一行；反向传播则会修改这张表，使向量逐渐适应预测任务。

例如 `input_ids=[[2,1,2]]` 时，输出依次是 `E[2]、E[1]、E[2]`，shape 为 `(1,3,D)`。第一个和第三个位置引用的是同一行权重，而不是为两次出现各造一张表。沿输入查表这条路径反向传播时，这两个位置给 `E[2]` 的梯度会相加。

## LM head 为每个 token 打分

Transformer 输出 `(B,T,D)` 的上下文向量，LM head 在每个位置上做一次 `D -> V` 线性投影。PyTorch 会把这层的权重存成 `(V,D)`，所以两端的 shape 对应为：

```text
token_embedding.weight: (V,D)
lm_head.weight:         (V,D)
logits:                 (B,T,V)
```

`logits[b,t,v]` 表示样本 `b` 在位置 `t` 对词表项 `v` 的未归一化分数。训练使用的 cross-entropy 会直接接收 logits，因为它内部已经包含数值更稳定的归一化计算，模型里不应提前再做一次 Softmax。

## 共享入口和出口的权重

Embedding 和 LM head 的权重恰好都是 `(V,D)`，Weight tying 利用了这个对称性，让两个模块引用同一个 Parameter：

```python
self.lm_head.weight = self.token_embedding.weight
```

同一张表在输入时按 id 取行，在输出时则让 hidden state 与每一行做内积，一次得到 `V` 个分数：

$$
H:(B,T,D),\qquad HE^\top:(B,T,V).
$$

若只在初始化时复制数值，代码看起来很像，含义却完全不同：

```python
self.lm_head.weight.data.copy_(self.token_embedding.weight.data)
```

这种写法会保留两份 Parameter，训练一开始就会各自更新。真正的共享可以用对象身份来区分：

```python
model.lm_head.weight is model.token_embedding.weight
```

Weight tying 是模型设计选择，不是 LM head 的必要条件。MiniMind 采用这种设计，因而不需要再为输出端保留一张独立的 `(V,D)` 权重表。

## `TinyLanguageModel` 只用来看入口和出口

`language_model.py` 里的 `TinyLanguageModel` 故意省略了 Attention，只把 token embedding 与一张可学习位置表相加，然后直接送入 LM head：

```text
token embedding + learned position embedding -> tied LM head
```

位置表可以让同一 token 在不同绝对位置产生不同分数，却不会把前文传给当前位置。因此，这个类适合核对查表、shape、LM head 和权重共享，不能用来判断上下文建模能力。

它接受已经错开一位的 `labels`，可以单独走通 loss 接口，但不会在 `forward` 中自动移动标签。带有 causal decoder blocks 和 RoPE 的整模见 [MiniMind Core](../task_27_minimind_core/README.md)，数据切片、PAD 与 loss mask 则由 [Next-token 训练](../task_28_next_token_training/README.md) 处理。

## 代码输出

在仓库根目录运行脚本，可以同时核对输出 shape 与 Parameter 共享：

```bash
python exercises/block_03_transformer/task_25_embedding_lm_head/language_model.py
```

正常输出中，`logits` 有三个维度，`weights shared` 则应为 `True`：

```text
logits: (1, 4, 40)
weights shared: True
```

Embedding 产生 `(B,T,D)`，LM head 接收同样宽度的 hidden states。中间还缺少一条让 token 互相读取的通路，[Attention 与 decoder-only](../task_20_transformer_theory/README.md) 从这个问题继续。

参考：[PyTorch：Automatic Differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)、[Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)。
