# Sampling

语言模型最后输出的是 logits.

你要把 logits 变成下一个 token.

选法不同, 生成风格会差很多.

## Greedy

每次选概率最大的 token:

```text
next = argmax(logits)
```

稳定, 但容易重复, 也容易无聊.

## Temperature

temperature 控制分布的尖锐程度:

$$
p = \mathrm{softmax}(logits / T)
$$

$T$ 小, 模型更保守.

$T$ 大, 模型更随机.

## Top-k

只保留概率最高的 k 个 token.

比如 `top_k=50`, 就只从 50 个候选里采样.

这能减少抽到离谱 token 的概率.

## Top-p

Top-p 也叫 nucleus sampling.

它不是固定保留 k 个 token, 而是保留累计概率达到 p 的那一小撮 token.

比如 `top_p=0.9`, 就保留概率从高到低累加到 0.9 的候选集合.

## 常见组合

实际生成时常见组合是:

```text
temperature = 0.7 或 1.0
top_p = 0.9
top_k = 40 或 50
```

没有永远最好的参数. 写故事、写代码、做问答, 需要的随机程度都不一样.
