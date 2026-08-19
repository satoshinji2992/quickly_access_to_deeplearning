const repo = "https://github.com/satoshinji2992/quickly_access_to_deeplearning";

const blocks = [
  { no: "01", kicker: "NUMPY FOUNDATIONS", title: "先把神经网络拆开看", text: "从一条直线走到 MNIST。前向传播、反向传播、Loss 和 Optimizer 都不藏在框架里。", points: ["线性回归", "圆形分类", "Mini DL Library", "MNIST MLP"], image: "/figures/block1_overview.png", href: `${repo}/blob/agent/refine-deep-learning-tutorial/chapters/01-%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86.md` },
  { no: "02", kicker: "VISION SYSTEMS", title: "再让像素流过 ResNet", text: "把卷积、im2col、BatchNorm 和残差连接连成一条可训练的 NumPy 图像分类管线。", points: ["CIFAR-100", "Conv2D", "Residual Block", "Checkpoint"], image: "/figures/block2_overview.png", href: `${repo}/blob/agent/refine-deep-learning-tutorial/chapters/02-ResNet%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB.md` },
  { no: "03", kicker: "SEQUENCE MODELS", title: "最后拼出一个小型 Transformer", text: "从位置编码和 causal attention 出发，补齐 GQA、SwiGLU、生成与逐层 KV Cache。", points: ["RoPE", "GQA", "MiniMind", "KV Cache"], image: "/figures/minimind_overview.png", href: `${repo}/blob/agent/refine-deep-learning-tutorial/chapters/03-Transformer%E4%B8%8EMiniMind.md` },
];

const figures = [
  ["IM2COL", "把滑动窗口变成一次矩阵乘法", "/figures/im2col_explained.png"],
  ["GROUPED QUERY ATTENTION", "4 个 Query 头，2 组共享 KV", "/figures/gqa.png"],
  ["KV CACHE", "只计算新 token，同时保持等价输出", "/figures/kv_cache.png"],
];

export default function Home() {
  return <main>
    <nav className="nav shell" aria-label="主导航">
      <a className="brand" href="#top" aria-label="返回首页"><span className="brand-mark">DL</span><span>DEEP LEARNING<br />FIELD NOTES</span></a>
      <div className="nav-links"><a href="#route">课程路线</a><a href="#figures">配图</a><a href="#reproduce">复现</a></div>
      <a className="nav-cta" href={repo}>GitHub ↗</a>
    </nav>

    <section id="top" className="hero shell">
      <div className="hero-copy">
        <div className="eyebrow"><span /> NUMPY · RESNET · TRANSFORMER</div>
        <h1>把深度学习<br /><em>真正跑通。</em></h1>
        <p className="hero-lead">一套从零实现的中文教程。不是 API 巡礼，而是从梯度、shape 和数据流出发，一路写到可训练的 ResNet 与带 KV Cache 的 decoder-only Transformer。</p>
        <div className="hero-actions"><a className="button primary" href="#route">开始探索 <span>↓</span></a><a className="button secondary" href={`${repo}#readme`}>阅读教程 <span>↗</span></a></div>
        <div className="hero-stats" aria-label="项目概览"><div><strong>15</strong><span>可运行主题</span></div><div><strong>03</strong><span>完整学习模块</span></div><div><strong>43</strong><span>自动检查</span></div></div>
      </div>
      <div className="hero-visual" aria-label="课程核心模型概览">
        <div className="paper-tag">PROJECT FIGURE 01</div>
        <img src="/figures/minimind_overview.png" alt="MiniMind 模型结构：从 token 到 decoder、logits 与 KV Cache" />
        <div className="visual-caption"><span>FIG. 01</span><p>Decoder-only Transformer 的训练与增量推理主线</p></div>
      </div>
    </section>

    <div className="ticker" aria-hidden="true"><div>FORWARD PASS <i>✦</i> BACKPROPAGATION <i>✦</i> IM2COL <i>✦</i> RESIDUAL CONNECTION <i>✦</i> ROPE <i>✦</i> GQA <i>✦</i> KV CACHE</div></div>

    <section id="route" className="route shell section">
      <header className="section-head"><div><span className="section-index">01</span><span className="eyebrow">LEARNING ROUTE</span></div><h2>三步，把黑盒<br />拆成看得见的计算。</h2><p>每一段都从小例子开始，用 shape、数值和测试把概念钉牢；读完说明，就能直接运行对应代码。</p></header>
      <div className="block-list">{blocks.map((block) => <article className="block-card" key={block.no}>
        <div className="block-meta"><span className="block-no">{block.no}</span><span>{block.kicker}</span></div>
        <div className="block-body"><h3>{block.title}</h3><p>{block.text}</p><div className="chips">{block.points.map((point) => <span key={point}>{point}</span>)}</div><a href={block.href}>进入这一章 <span>↗</span></a></div>
        <div className="block-image"><img src={block.image} alt={`${block.title}概览图`} /></div>
      </article>)}</div>
    </section>

    <section id="figures" className="figures section"><div className="shell">
      <header className="section-head light"><div><span className="section-index">02</span><span className="eyebrow">VISUAL EXPLANATIONS</span></div><h2>复杂结构，<br />尽量一眼看懂。</h2><p>矩阵值、张量 shape、连接方向和 head 映射都经过核对。图不是装饰，而是正文的一部分。</p></header>
      <div className="figure-grid">{figures.map(([label, title, image], index) => <figure className={index === 0 ? "figure-card featured" : "figure-card"} key={label}><div className="figure-frame"><img src={image} alt={title} /></div><figcaption><span>0{index + 1} / {label}</span><strong>{title}</strong></figcaption></figure>)}</div>
    </div></section>

    <section id="reproduce" className="reproduce shell section">
      <div className="reproduce-copy"><span className="section-index">03</span><div className="eyebrow">REPRODUCIBLE BY DESIGN</div><h2>不止“能运行”，<br />还要算得对。</h2><p>数据隔离、有限差分梯度、causal 性质、单 batch 拟合和 checkpoint round-trip 都有对应检查。失败时，能快速知道问题落在哪一层。</p><a className="button primary" href={`${repo}/tree/agent/refine-deep-learning-tutorial/tests`}>查看测试 <span>↗</span></a></div>
      <div className="terminal" role="img" aria-label="测试全部通过的终端示例"><div className="terminal-bar"><i /><i /><i /><span>quickly_access_to_deeplearning — tests</span></div><pre><span className="muted">$</span> python -m unittest discover -s tests -v{"\n\n"}<span className="ok">✓</span> circle labels &amp; split isolation{"\n"}<span className="ok">✓</span> conv2d finite-difference gradient{"\n"}<span className="ok">✓</span> residual identity / projection{"\n"}<span className="ok">✓</span> causal attention property{"\n"}<span className="ok">✓</span> checkpoint round-trip{"\n\n"}<strong>Ran 43 checks</strong>{"\n"}<span className="success">OK</span></pre></div>
    </section>

    <section className="closing"><div className="shell closing-inner"><div className="eyebrow"><span /> OPEN SOURCE · CHINESE TUTORIAL</div><h2>从 <code>y = ax + b</code><br />到 <code>token → logits</code></h2><p>代码、讲解、配图和参考实现都在同一个仓库里。</p><a className="button paper" href={repo}>打开 GitHub 仓库 <span>↗</span></a></div></section>
    <footer className="footer shell"><div className="brand"><span className="brand-mark">DL</span><span>DEEP LEARNING<br />FIELD NOTES</span></div><p>从零实现 · 数值核对 · 可复现</p><a href="#top">BACK TO TOP ↑</a></footer>
  </main>;
}
