# Next-token 训练

[MiniMind Core](../task_27_minimind_core/README.md) 已经能从 token ids 算出 logits，训练脚本还差一段数据管道：原始文本要编成 id，切出输入和下一 token 标签，再和 mask 一起送入模型。`train.py` 内置了一段 UTF-8 中文记录，也接受 `--text` 指向本地文件。例子不需联网，字符 Tokenizer 的行为也可以直接从代码中查看。

![Next-token 标签、loss mask 与 causal/padding mask](assets/shifted_labels.png)

<div class="widget-mount" data-widget="next-token"></div>

## 先切文本，再建词表

代码先按原来的先后顺序把文本分成两段，分界取在约 85% 处；附近有换行时，就在换行处结束训练文本，使验证部分从一条完整记录开始。单行文本没有合适的换行，才直接按字符位置切开：

```text
前约 85% -> train text
后约 15% -> validation text
```

两个 split 因而不会复用同一个字符位置。字符 Tokenizer 只在 train text 上建词表，validation 中从未在训练部分出现的字符会被编成 `<unk>`。词表还保留了四个特殊 token，各自处理不同的边界情况：

```text
<pad>  batch 补齐
<unk>  词表外字符
<bos>  序列开始
<eos>  序列结束
```

例如训练文本只有“小猫睡觉”，词表就能区分这四个字；验证文本里的“小狗睡觉”会变成“小、`<unk>`、睡、觉”。模型此时既无法看见“狗”的身份，也不可能从 `<unk>` 恢复原字。脚本打印 `val_unknown_chars`，就是为了把这种词表覆盖不足和模型预测不好分开看。内置短文以最后一条“周四早晨”的记录作验证，35 个字符都在前面的训练记录中出现过，但两个完整句子没有在训练中复用，因此输出是 `val_unknown_chars=0/35`。

按字符切分的好处是 id 与原文容易对照，代价是同一个词可能占据多个位置，也无法代表实用 LLM 常见的 BPE 或 SentencePiece 处理。这里相邻几条气象记录的写法很接近，验证主要观察已见字符在新组合里的预测；处理更大的语料时，通常会按文档划分 split，并在分割前去重，避免同源文本同时出现在训练与验证中。

## 标签就是同一段 id 向左错一位

`NextTokenDataset` 每次取出 `seq_len+1` 个连续 id，前 `seq_len` 个作为输入，从第二个开始的 `seq_len` 个作为标签，于是每个位置都对应着它的下一个 token：

```text
block: [t0, t1, t2, t3, t4]
input: [t0, t1, t2, t3]
label: [t1, t2, t3, t4]
```

Tokenizer 只在完整的 train/validation split 开头加一个 BOS，末尾加一个 EOS，不会给每个短 block 重复包边界符。所以只有第一块可能以 BOS 开头，只有最后一块可能包含 EOS，中间块是原文中连续的一段。切片起点按 `0, seq_len, 2*seq_len, ...` 前进，相邻 block 共享一个边界 token：它在前一块是最后一个 label，在后一块是第一个 input。这样，每条相邻 token 转移恰好训练一次。

## Padding 同时影响 Attention 和 loss

文本末尾的 block 往往不足 `seq_len`，Dataset 会在右侧补 PAD，并返回三个 shape 对齐的张量：

```text
input_ids:      (B,T)
labels:         (B,T)
attention_mask: (B,T), True 表示有效输入
```

Causal mask 处理时间方向，禁止读取未来；padding mask 处理 batch 补齐，禁止将 PAD 当成 key。Loss 端并没有从 Dataset 取出第四个张量，而是在 `MiniMindCore.forward` 中现场派生有效 target：

```python
valid_target = attention_mask & (labels != pad_token_id)
targets[~valid_target] = -100
```

Cross-entropy 忽略 `-100`，这就是图中 loss mask 的实际去向。例如 EOS 对应的 input 位置仍是有效 token，所以 attention mask 为 1；它的下一个 label 已是 PAD，该 target 会被改为 `-100`。

把最后一个短 block 摊开看会更清楚。假设只剩“觉”和 EOS，`seq_len=4`，补齐后是：

```text
input_ids       [觉,  EOS, PAD, PAD]
labels          [EOS, PAD, PAD, PAD]
attention_mask  [ 1,   1,   0,   0]
有效 target      [ 1,   0,   0,   0]
```

这里只有“读到觉，预测 EOS”这一项产生损失。EOS 的输入位置仍能读取前缀，但没有需要预测的真实字符；不能因为它的 attention mask 是 1，就把后面的 PAD 也算作答案。当前 batch 的 loss 是有效项的平均，一行全是 PAD 时不额外增加分母。

## 一次参数更新

训练 loader 会打乱 block，验证 loader 保持固定顺序。数据进入模型后，一次参数更新仍然沿用前两章的基本顺序：

```python
logits, loss = model(input_ids, labels, attention_mask=attention_mask)
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

前两章自己写的 `backward`，在这里由 PyTorch 根据 forward 中的运算自动构造。`loss.backward()` 把梯度累积到各 Parameter 的 `.grad`，尚未改变参数；真正的更新仍在 `optimizer.step()` 中。每一批前清掉上一批梯度，是因为 PyTorch 默认会累加梯度，而当前脚本没有做多批次梯度累积。

默认模型只有 `D=64`、2 个 blocks、4 个 query heads 和 2 个 KV heads，SwiGLU 中间宽度为 128。这个尺寸是为了在 CPU 上看清从数据到更新的全过程，生成文本会很粗糙。梯度总范数超过 1.0 时，梯度裁剪会按比例缩小所有梯度，避免单次更新过大；它不会修复错误的 mask 或标签。

## Validation loss 按有效 token 加权

模型返回的 loss 是当前 batch 内有效 target token 的平均。验证集最后一批通常含有更多 PAD，若直接对各 batch loss 再取一次平均，token 较少的批次就会获得过大权重。`evaluate` 因而按有效 target 数量做加权：

```text
sum(batch_loss * valid_target_count) / sum(valid_target_count)
```

例如一批有 100 个有效 token、平均 loss 为 2，另一批只有 10 个、平均 loss 为 4，合在一起应为 $(100\times2+10\times4)/110\approx2.18$，不能把两个均值直接平均成 3。由于同一篇文本切成不同 batch 后总损失不应改变，测试还会把这个加权结果与全部有效 token 一次计算的结果对照。

脚本会先记录训练前的验证值，随后按 `--eval-every` 定期验证，并在最后一步再验证。`evaluate` 临时切到 `eval()` 并关闭梯度记录，结束后恢复原来的模式，接着训练。`eval()` 决定 Dropout 等层的行为，`torch.no_grad()` 决定是否记录反向传播所需的信息，两者各管一件事。

以默认语料、CPU、seed 0 运行 80 步，可以看到类似下面的变化，浮点尾数会随环境略变：

```text
step=000 validation_loss=4.7652
step=020 train_loss=2.5011  validation_loss=3.2148
step=040 train_loss=0.8648  validation_loss=2.4143
step=060 train_loss=0.2314  validation_loss=2.2867
step=080 train_loss=0.0866  validation_loss=2.4035
```

开始时两个 loss 都下降，说明训练中出现的写法也帮助了新句子的预测；到后面，训练 loss 继续下降，验证 loss 却开始回升，模型对这几块训练文本的记忆已经比对新句子的改进更明显。这个变化比只看最后一个训练 loss 更有用，不过验证只有一条短记录，曲线也容易受具体措辞影响，不能代表通用语言能力。`train_loss` 是最近一次更新前的单 batch 值，`validation_loss` 则是在更新后对整个验证 split 重新计算，两者也不是同一时刻、同一组样本的测量。

## Checkpoint 保存模型之外的状态

Checkpoint 不只保存模型权重，还要保留恢复数据解码和优化过程所需的信息：

```text
config
model_state
optimizer_state
tokenizer
step
val_loss
training_state   # 语料指纹、数据切片设置、shuffle seed、随机数状态
```

`load_checkpoint` 先使用 config 重建模型，再载入参数和 Tokenizer。加载前后的 eval logits 应在浮点容差内一致，Embedding/LM head 的 Parameter 共享也会保留。只做生成时读到这里就够了；继续训练还需要恢复 AdamW 的历史动量，以及下一步该读哪个 batch。

`--resume` 会恢复这些状态。每轮数据遍历使用 `seed + epoch` 决定 shuffle 顺序，再根据已完成的 step 找到该轮里尚未读取的 batch。这样，在同一 CPU 环境中连续训练 7 步，与先训练 3 步、保存后再训练 4 步，得到的参数可以逐元素一致；跨设备或不同软件版本不要求这一点。为避免把另一份数据误当成上次训练，脚本会核对语料指纹、`seq_len` 和 batch size。

指定 `--checkpoint` 的文件会在每次验证后保存最近状态，它不是自动挑选的“最好模型”。想比较不同训练长度时，可以分别写到不同路径。保存时先写入同目录临时文件，写完才替换旧文件；如果中途写入失败，上一次保存的模型仍然可以读取。旧版本只保存模型和优化器、没有 `training_state` 的文件仍能用于生成，但不能还原新的数据遍历状态，`--resume` 会对此给出提示。

## 运行训练示例

要快速看一遍数据、模型、损失和 checkpoint 是否连通，可以只跑四步：

```bash
python exercises/block_03_transformer/task_28_next_token_training/train.py \
  --steps 4 --seq-len 24 --batch-size 2 \
  --checkpoint /tmp/minimind_smoke.pt
```

终端会打印 device、语料字符数、词表大小、验证集未知字符数、各阶段 loss 和 checkpoint 路径。如果希望在同一份内置语料上多训练一会，可以把步数提到 80：

```bash
python exercises/block_03_transformer/task_28_next_token_training/train.py \
  --steps 80 --eval-every 20 --checkpoint /tmp/minimind_demo.pt
```

`--text` 可以将内置短文替换为本地 UTF-8 文件：

```bash
python exercises/block_03_transformer/task_28_next_token_training/train.py \
  --text path/to/corpus.txt --steps 200 \
  --checkpoint /tmp/minimind_demo.pt
```

文本过短时，train/validation 两侧无法各自组成 next-token 对，脚本会直接报错。成功训练后，checkpoint 中已有词表和模型参数，[自回归生成](../task_29_generate_sampling/README.md) 会从这个文件恢复模型，并从最后一位 logits 开始选 token。

接着内置语料的 80 步继续训练，可以另存一个文件，保留之前的模型以便比较：

```bash
python exercises/block_03_transformer/task_28_next_token_training/train.py \
  --resume /tmp/minimind_demo.pt --steps 20 \
  --checkpoint /tmp/minimind_step100.pt
```

`--steps 20` 表示再更新 20 次，因此最后打印 `step=100`。原来用 `--text` 训练时，续跑也要传入同一份文件；模型尺寸、词表、学习率状态、`seq_len` 和 batch size 都从 checkpoint 恢复。

保存和恢复的基本接口可对照 [PyTorch：Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)，损失的忽略规则见 [`cross_entropy` 的 `ignore_index`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)。
