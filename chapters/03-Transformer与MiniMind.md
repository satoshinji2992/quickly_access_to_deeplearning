# Transformer 语言模型：从文本到下一个 token

前两章的模型一次输出一个数或一个类别。语言模型反复回答的其实是同一个问题：读到一段文本后，下一个 token 会是什么？例如读到“小猫睡”，模型会给“觉”“了”“着”等候选项打分。这些分数不能只看末尾的“睡”，前面的“小猫”也得进入预测。

token 是文本切分后的单位。本章的小模型按字符切分，“小猫睡觉”会变成“小、猫、睡、觉”；实际的大模型通常会混合使用字、词和词片段。切分方式会影响序列长度，但后面的数据流相同：token 先变成 id，再变成向量。

实现从这一章换成 PyTorch。前两章算过的梯度并没有换一种原理，只是 `loss.backward()` 会替我们沿 forward 中的运算自动求导；参数仍要由优化器更新。Attention 的投影、矩阵乘法和 mask 会逐项写出来，不直接调用现成的 Attention 层。[Embedding 页](../exercises/block_03_transformer/task_25_embedding_lm_head/README.md) 先解释代码中的 `nn.Module` 和 Parameter，再接到下面的向量计算。

## 文本如何进入模型

Tokenizer 为词表中的每个 token 分配一个整数 id，id 只是查表地址，编号 12 并不比编号 6 “大一倍”。Embedding 则是一张可训练的表，每行存放一个 token 的向量。为了让整个 Attention 过程可以手算，我们只给每个 token 两个特征：

```text
token      id       embedding
小         12       [1, 0]
猫         37       [0, 1]
睡         81       [1, 1]
```

按照序列顺序把三个向量排成行，“小猫睡”就对应输入矩阵

$$
X=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}.
$$

真实的 embedding 会随训练改变，这组数只用来跟踪后续计算。成批输入时，id 张量的 shape 是 `(B,T)`，查表后变为 `(B,T,D)`，其中 `B` 是样本数，`T` 是序列长度，`D` 是向量宽度。

<div class="widget-mount" data-widget="token-embed-3d"></div>

如果同一个 MLP 分别处理每一行，“睡”所在位置仍然不知道前面出现了“小猫”。Attention 要解决的就是这个问题：让一行向其他行取回信息。

## 手算一次 Attention

输入 $X$ 分别经过三个线性变换：

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

Q（Query）说明当前位置想找什么，K（Key）用来接受匹配，V（Value）则是匹配后取回的内容。名字只描述计算分工，不是人工标注的语法角色。手算时取 $W_Q=W_K=W_V=I$，所以 Q、K、V 都等于上面的 $X$；真实模型会学习三组不同的投影。

最后一个位置的 query 是 $q=[1,1]$，它与三个 key 的点积为 `[1,1,2]`。每个向量有两维，所以再除以 $\sqrt2$，得到缩放后的分数

$$s=[0.7071,\ 0.7071,\ 1.4142].$$

这里除以 $\sqrt2$ 是为了控制分数的尺度。向量变宽后，点积里相加的项也变多；若各维近似独立、均值为 0、方差为 1，含 $d$ 项的点积方差约为 $d$，除以 $\sqrt d$ 后才回到约 1。否则即使还没学到合适的匹配，较大的分数也容易让 Softmax 过早接近只选一个位置。

Softmax 把这三个分数变成和为 1 的读取权重

$$a\approx[0.2483,\ 0.2483,\ 0.5035],$$

再用权重汇总同一组 V：

$$
o=0.2483[1,0]+0.2483[0,1]+0.5035[1,1]
\approx[0.7518,0.7518].
$$

最后位置原来的向量是 `[1,1]`，新向量已经混入前两个位置的内容。0.2483 和 0.5035 是读取各个位置的比例，不是下一个 token 的概率。把所有 query 放回矩阵中同时计算，就得到常见的写法：

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_{head}}}+M\right),
\qquad O=AV.
$$

`QKᵀ` 的最后两维是 `(T,T)`，第 $t$ 行记录位置 $t$ 如何读取各个 key。$M$ 是能广播到 score shape 的 mask，用来限制哪些位置允许被读取。

<div class="widget-mount" data-widget="qkv-flow"></div>

## 训练时不让答案提前泄露

把“小猫睡觉”改造成 next-token 数据时，输入与标签错开一位，BOS 表示序列开始：

```text
输入：[BOS, 小, 猫, 睡]
标签：[小,  猫, 睡, 觉]
```

模型在“猫”的位置应该预测“睡”，可整段序列在训练时已经放进同一个张量。若该位置可以读取右边的“睡”，它就是在抄答案，即使损失很低也不代表学会了预测。Causal mask 因而将 query $t$ 的可见范围限制为 key $0\ldots t$：

```text
        key 0  key 1  key 2  key 3
query 0   ✓      ×      ×      ×
query 1   ✓      ✓      ×      ×
query 2   ✓      ✓      ✓      ×
query 3   ✓      ✓      ✓      ✓
```

允许读取的位置在 $M$ 中取 0，禁止的位置取负无穷，经过 Softmax 后权重就是 0。修改未来 token 不会改变前缀位置的输出，改动前文却可以影响后面的位置。

<div class="widget-mount" data-widget="causal-mask"></div>

## 顺序从哪里来

如果没有位置信号，也没有 causal mask 这类非对称约束，Self-attention 只会根据向量内容匹配；调换 token 顺序只会同样调换输出。Causal mask 表达了“左边可见、右边不可见”，却没有直接告诉模型绝对位置或两个 token 相隔多远。

位置信息可以用可学习的位置表表示，也可以把固定的正弦位置编码加到 token embedding 上。本仓库的 MiniMind 使用 RoPE：embedding 本身不变，Q/K 的成对分量根据位置旋转，使位置差进入匹配分数。

单个 head 只产生一组读取权重，Multi-head attention 用多组投影分别计算，再把各组输出拼回 `(B,T,D)`。MiniMind 采用的 GQA 继续保留多个 query heads，但让几个 query heads 共享一组 K/V，以减少 K/V 参数和后面的缓存量。

## Attention 之外的逐位置计算

Attention 负责混合不同位置的信息，FFN 则在每个位置内独立变换特征，不沿序列轴求和。在 Pre-Norm decoder block 中，两个子层通过残差连接串在一起：

```text
u = x + Attention(RMSNorm(x))
h = u + FFN(RMSNorm(u))
```

残差主路为输入和梯度保留了直接通道，RMSNorm 在进入子层前控制特征尺度。两条支路都要输出 `(B,T,D)`，否则无法和主路逐元素相加。

MiniMind 的 FFN 使用 SwiGLU：输入被投影为两路，`SiLU(gate) * up` 完成逐元素调制，再投影回 `D`。SiLU 的输出可以为负，也可以大于 1；“门”是对乘法分支的形象称呼，不是概率开关。

## 从 hidden states 到训练损失

经过多个 decoder blocks 后，LM head 会为词表中的每个 token 产生一个 logit，于是张量由 `(B,T,D)` 变为 `(B,T,V)`。字母 `V` 在此表示词表大小，与 Attention 里的 Value 只是碰巧重名。

假设最后位置对“觉、了、着”给出 logits `[2,1,0]`，Softmax 后的概率约为 `[0.6652,0.2447,0.0900]`；真实答案是“觉”时，该位置的交叉熵为 $-\log(0.6652)\approx0.4076$。这里的 Softmax 沿词表轴归一化，在候选 token 之间形成概率分布；Attention 中的 Softmax 则沿 key 轴归一化，在可读取的位置之间分配权重。

输入和标签错开一位后，一次 forward 就能算出每个位置的损失。训练数据已经给出各位置的真实前文，所以这些位置可以并行计算，causal mask 则保证每个位置都看不到自己的答案。

![输入与标签错开一位](../assets/images/shifted_labels.png)

组 batch 时补入的 PAD 不是正常标签，既不应计入损失，也不能成为其他位置读取的上下文。[训练页](../exercises/block_03_transformer/task_28_next_token_training/README.md) 会区分两种 mask：`attention_mask: (B,T)` 传给 Attention；损失端不额外存一张 `loss_mask`，而是由有效输入位置与 `labels != PAD` 现场得到，并将无效 target 改为 `-100`。

## 生成时只能一步一步来

训练数据包含每个位置的真实前文，而生成时的后续 token 还不存在。模型只能读取当前前缀，从最后位置的 logits 中选出一个 token，把它接回输入后再继续：

```text
“小猫睡”   -> 选出“觉”
“小猫睡觉” -> 再预测一个 token
```

直接取最大 logit 叫 greedy decoding，按概率抽取则叫 sampling。Temperature 改变分布的尖锐程度，top-k 只保留分数最高的 $k$ 个候选，top-p 保留概率从高到低累加后达到阈值的最小前缀。它们改变的只是从同一组 logits 中怎样选 token，不会补上模型本身没学到的内容。

<div class="widget-mount" data-widget="sampling"></div>

生成必须等待上一步选出新 token，但不必每轮都重算旧前缀的全部中间结果。KV Cache 会保存各层旧位置的 K/V，每轮只投影新 token；新 query 仍然要与所有可见 key 计算匹配分数。

第一次跑这套代码时，用内置短文训练几十步就能保存 checkpoint，再输入“周一早晨，”观察生成结果。[训练页](../exercises/block_03_transformer/task_28_next_token_training/README.md) 会把训练 loss 与独立文本上的验证 loss 放在一起看：记住这段短文通常很快，但它不等于学会继续陌生文本。字符词表没覆盖到的字还会变成 `<unk>`，这也是读验证数字前需要先核对的事。

## MiniMind 的完整数据流

![MiniMind 的完整结构与推理路径](../assets/images/minimind_overview.png)

`MiniMindCore` 使用 RoPE、GQA、Pre-RMSNorm 和 SwiGLU，并让输入 Embedding 与 LM head 共享同一张权重表。训练和生成共用模型主干，但数据流不同：

```text
原始 token ids
  ├─> input_ids ─> embedding ─> causal decoder blocks ─> logits ─┐
  └─> 向左错一位的 labels ─> 有效 target mask ───────────────┘
                                                        │
                                                masked cross-entropy

生成：当前前缀 ─> 模型 ─> logits[:, -1] ─> 选 token ─> 接回前缀
```

[入口与出口](../exercises/block_03_transformer/task_25_embedding_lm_head/README.md) 先说 token id 如何进入向量空间，以及 hidden state 如何变回词表 logits。[Attention 与 decoder-only](../exercises/block_03_transformer/task_20_transformer_theory/README.md) 随后定义 token 之间的信息通路；[正弦位置编码](../exercises/block_03_transformer/task_21_sinusoidal_position/README.md) 和 [RoPE](../exercises/block_03_transformer/task_22_rope_position/README.md) 处理顺序，[Causal Attention 与 GQA](../exercises/block_03_transformer/task_23_causal_attention/README.md) 把投影、mask 与多头计算接起来。

[SwiGLU](../exercises/block_03_transformer/task_24_swiglu_ffn/README.md) 补上逐位置的特征变换，[Decoder Block](../exercises/block_03_transformer/task_26_decoder_blocks/README.md) 再用 RMSNorm 和残差把 Attention 与 FFN 组装起来。

再往下四页会把模型、训练、生成和缓存放在一起运行：[MiniMind Core](../exercises/block_03_transformer/task_27_minimind_core/README.md)、[Next-token 训练](../exercises/block_03_transformer/task_28_next_token_training/README.md)、[自回归生成](../exercises/block_03_transformer/task_29_generate_sampling/README.md) 和 [KV Cache](../exercises/block_03_transformer/task_30_kv_cache/README.md)。
