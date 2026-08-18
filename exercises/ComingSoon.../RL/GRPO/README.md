# GRPO

GRPO（Group Relative Policy Optimization）对同一 prompt 采样一组回答，再用组内 reward 的相对值构造 advantage。这样可以不训练一个与 policy 同规模的 value model。

```text
one prompt -> G completions -> G rewards -> group-relative advantages
```

组内标准化并不会自动解决 reward hacking，还会受组大小、reward 并列、全部答错或全部答对等边界情况影响。复现实验时需要记录采样策略、reward 定义、KL 项与每个 prompt 的有效样本数。

本页尚未提供训练实现。GRPO 的原始描述见 [DeepSeekMath](https://arxiv.org/abs/2402.03300)。
