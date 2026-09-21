# Next-token 训练

[MiniMind Core](../task_27_minimind_core/README.md) 已经能从 token ids 算出 logits，训练脚本还差一段数据管道：原始文本要编成 id，切出输入和下一 token 标签，再和 mask 一起送入模型。`train.py` 内置了一段 UTF-8 中文记录，也接受 `--text` 指向本地文件。例子不需联网，字符 Tokenizer 的行为也可以直接从代码中查看。

![Next-token 标签、loss mask 与 causal/padding mask](assets/shifted_labels.png)

<div class="widget-mount" data-widget="next-token"></div>

## 先切文本，再建词表

代码先按字符位置将连续文本分成两段，并保持原来的先后顺序：

```text
前 85% -> train text
后 15% -> validation text
```

两个 split 因而不会复用同一个字符位置。字符 Tokenizer 只在 train text 上建词表，validation 中从未在训练部分出现的字符会被编成 `<unk>`。词表还保留了四个特殊 token，各自处理不同的边界情况：

```text
<pad>  batch 补齐
<unk>  词表外字符
<bos>  序列开始
<eos>  序列结束
```

按字符切分的好处是 id 与原文容易对照，代价是同一个词可能占据多个位置，也无法代表实用 LLM 常见的 BPE 或 SentencePiece 处理。直接按字符位置分割连续文本同样只适合这个小例子；处理真实语料时，通常会按文档划分 split，并在分割前去重，避免同源文本同时出现在训练与验证中。

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

## 一次参数更新

训练 loader 会打乱 block，验证 loader 保持固定顺序。数据进入模型后，一次参数更新仍然沿用前两章的基本顺序：

```text
forward -> masked cross-entropy
zero_grad -> backward -> gradient clipping -> AdamW.step
```

默认模型只有 `D=64`、2 个 blocks、4 个 query heads 和 2 个 KV heads，SwiGLU 中间宽度为 128。这个尺寸是为了在 CPU 上看清从数据到更新的全过程，生成文本会很粗糙。梯度总范数超过 1.0 时，梯度裁剪会按比例缩小所有梯度，避免单次更新过大；它不会修复错误的 mask 或标签。

## Validation loss 按有效 token 加权

模型返回的 loss 是当前 batch 内有效 target token 的平均。验证集最后一批通常含有更多 PAD，若直接对各 batch loss 再取一次平均，token 较少的批次就会获得过大权重。`evaluate` 因而按有效 target 数量做加权：

```text
sum(batch_loss * valid_target_count) / sum(valid_target_count)
```

验证时模型处于 `eval()` 模式，并关闭梯度记录。当前脚本只在训练结束后计算一次 validation loss，没有 early stopping 或 best-checkpoint 选择；几十步、小语料上的数字只能说明数据、模型与损失能够一起运行，不能证明模型具备通用语言能力。

## Checkpoint 保存模型之外的状态

Checkpoint 不只保存模型权重，还要保留恢复数据解码和优化过程所需的信息：

```text
config
model_state
optimizer_state
tokenizer
step
val_loss
```

`load_checkpoint` 先使用 config 重建模型，再载入参数和 Tokenizer。加载前后的 eval logits 应在浮点容差内一致，Embedding/LM head 的 Parameter 共享也会保留。CLI 暂时没有 `--resume`，因而文件虽然包含 `optimizer_state`，继续训练时仍需创建 optimizer，并显式调用 `load_state_dict`。

## 运行训练示例

要快速看一遍数据、模型、损失和 checkpoint 是否连通，可以只跑四步：

```bash
python exercises/block_03_transformer/task_28_next_token_training/train.py \
  --steps 4 --seq-len 24 --batch-size 2 \
  --checkpoint /tmp/minimind_smoke.pt
```

终端会打印 device、语料字符数、各阶段 loss、validation loss 和 checkpoint 路径。如果希望在同一份内置语料上多训练一会，可以把步数提到 80：

```bash
python exercises/block_03_transformer/task_28_next_token_training/train.py \
  --steps 80 --checkpoint /tmp/minimind_demo.pt
```

`--text` 可以将内置短文替换为本地 UTF-8 文件：

```bash
python exercises/block_03_transformer/task_28_next_token_training/train.py \
  --text path/to/corpus.txt --steps 200 \
  --checkpoint /tmp/minimind_demo.pt
```

文本过短时，train/validation 两侧无法各自组成 next-token 对，脚本会直接报错。成功训练后，checkpoint 中已有词表和模型参数，[自回归生成](../task_29_generate_sampling/README.md) 会从这个文件恢复模型，并从最后一位 logits 开始选 token。
