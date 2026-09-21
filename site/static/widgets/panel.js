/* Interactive widgets: registry + inline lazy mounting in the article flow.
 *
 * Contract for widget files (w-<id>.js):
 *   DLWidget.register(id, mountFn)
 *     - mountFn(container) is called once when the widget scrolls near the
 *       viewport; it builds its DOM inside and returns an optional dispose().
 *   Docs embed <div class="widget-mount" data-widget="<id>">.
 */
(function () {
  'use strict';

  var REGISTRY = {
    'linear-fit': { file: 'w-linear-fit.js', title: '线性回归：调 a 与 b' },
    'conv-sweep': { file: 'w-conv-sweep.js', title: '卷积：拖动核扫过图像' },
    'kv-cache': { file: 'w-kv-cache.js', title: 'KV Cache：逐步解码' },
    'mlp-boundary': { file: 'w-mlp-boundary.js', title: 'MLP：权重与决策边界' },
    'backprop-steps': { file: 'w-backprop-steps.js', title: '反向传播：逐步走一遍' },
    'optimizer-race': { file: 'w-optimizer-race.js', title: '优化器：同一起点赛跑' },
    'norm-playground': { file: 'w-norm-playground.js', title: 'BatchNorm 与 LayerNorm' },
    'image-batch': { file: 'w-image-batch.js', title: '一张图片的 NCHW' },
    'residual-block': { file: 'w-residual.js', title: '残差块：两分支与梯度直通路' },
    'pool-sweep': { file: 'w-pool-sweep.js', title: '池化：窗口扫描与梯度路由' },
    'im2col-grad': { file: 'w-im2col-grad.js', title: 'im2col 与梯度回传' },
    'causal-mask': { file: 'w-causal-mask.js', title: '因果掩码：训练与推理' },
    'qkv-flow': { file: 'w-qkv-flow.js', title: 'Q/K/V 计算分步' },
    'pos-encoding': { file: 'w-pos-encoding.js', title: '正弦位置编码：多尺度频率' },
    'rope-rotate': { file: 'w-rope-rotate.js', title: 'RoPE：位置变成旋转角' },
    'gqa-map': { file: 'w-gqa-map.js', title: 'GQA：分组共享 KV' },
    'token-embed-3d': { file: 'w-token-embed-3d.js', title: 'Token 向量的三维投影' },
    'sampling': { file: 'w-sampling.js', title: '采样：温度与截断' },
    'swiglu-gate': { file: 'w-swiglu-gate.js', title: 'SwiGLU：门控与参数量' },
    'next-token': { file: 'w-next-token.js', title: 'Next-token：标签错位与双 mask' },
  };

  var GUIDES = {
  "linear-fit": [
    "只挪动截距，能让直线贴近所有点吗？",
    [
      "先只拖动 b，观察直线平移与红色误差线。",
      "重置后选“只更新 b”，点一步；再重置并选“同时更新”，比较损失。",
      "从相同起点比较学习率 0.12 与 1，观察是否越过低点。"
    ],
    "b 只能平移直线，无法修正斜率。梯度在当前位置决定小幅调整的方向；过大的学习率会增大损失。这里只拟合列出的教学数据，不代表薪资预测效果。"
  ],
  "mlp-boundary": [
    "改变一条连接，为什么多个位置的分类会变化？",
    [
      "先观察初始分类边界与样本点。",
      "选择并调整一个权重，比较边界哪些区域改变。",
      "结合正文的四个 ReLU 例子，区分一个神经元与整张边界。"
    ],
    "参数参与一批位置的共同计算，因此一个参数可以影响许多点。ReLU 的分段行为经多层组合后产生复杂边界；单独一个 ReLU 不会自动画出圆。"
  ],
  "backprop-steps": [
    "某个参数的梯度从哪里来？",
    [
      "用“下一步”看到损失，先确认预测与标签。",
      "继续追踪一个权重：它乘了哪个输入，又收到多少上游梯度？",
      "到更新一步时，对照更新前后的损失。"
    ],
    "权重梯度收集“输入 × 上游梯度”的贡献，ReLU 还会遮断负输入处的路径。一步降低一个样本的损失，不等于已经学会整类数据。"
  ],
  "image-batch": [
    "一张彩色图的三个通道存在哪里？",
    [
      "点一张缩略图，查看 R、G、B 三个通道。",
      "在同一像素位置比较三个通道的数值。",
      "再换一张图片，分清批量轴和颜色轴。"
    ],
    "一个像素对应三个颜色数值；换图片改变 N，换颜色改变 C。NCHW 的轴顺序是图片、通道、高、宽。"
  ],
  "conv-sweep": [
    "一个特征图格子由哪九个位置算出？",
    [
      "先把窗口停在一个位置，逐项对照输入与核。",
      "向右移动一次，找出窗口新增和离开的像素。",
      "观察输出位置改变，而核的同一组权重仍被复用。"
    ],
    "每个输出是一个窗口的加权和。同一核在各处复用，形成局部连接与参数共享；演示指定的核不是训练后必然出现的结果。"
  ],
  "pool-sweep": [
    "同一窗口取最大值和平均值，信息怎样不同？",
    [
      "先观察一个窗口，比较 MaxPool 与 AvgPool。",
      "切到反向视图，比较梯度送到哪些格子。"
    ],
    "MaxPool 将梯度交给选中的最大值，AvgPool 在窗口中均分。池化汇总信息，通常无法从输出还原整个输入窗口。"
  ],
  "residual-block": [
    "主分支输出接近零，原输入还能传下去吗？",
    [
      "先看 identity 模式，逐步走到相加节点。",
      "切换 projection，检查两分支相加前的尺寸。",
      "观察下方简化梯度图，再核对它的假设。"
    ],
    "identity 分支直接保留输入，projection 用变换对齐尺寸。简化图只说明额外路径；真实网络的分支导数可以抵消，残差不保证梯度下界。"
  ],
  "im2col-grad": [
    "同一个像素出现在几个窗口里？",
    [
      "前向点击一个输出，查看对应窗口行与核列。",
      "切换反向，追踪中间像素收到的多个贡献。"
    ],
    "窗口展开会重复引用原像素。反向要把这些贡献相加，不能把窗口表当作没有重叠的数组直接 reshape 回去。"
  ],
  "qkv-flow": [
    "读取位置的权重，怎样变成一个输出向量？",
    [
      "分步前进到 scores，点一个格子看 Q 行与 K 行的点积。",
      "继续看遮挡与 Softmax，核对每行权重和为 1。",
      "最后对照一行权重怎样乘到各行 V 上。"
    ],
    "score 是匹配分数，Softmax 后才是读取权重，输出是 V 的加权和。这些权重在位置之间分配，不是词表里的下一 token 概率。"
  ],
  "causal-mask": [
    "位置 2 能读取位置 3 的答案吗？",
    [
      "对照训练输入和错开一位的目标。",
      "观察每个 query 对应的允许区域，再比较逐步推理。"
    ],
    "位置 2 只读取 0 到 2；因此改变位置 3 不应影响它。训练已有真实前文可并行，生成则依赖刚选出的 token。"
  ],
  "token-embed-3d": [
    "屏幕上的三个坐标，是模型的全部特征吗？",
    [
      "选择一个 token，比较三维坐标与原始八维向量。",
      "旋转视图，区分屏幕位置改变与原始向量改变。"
    ],
    "投影只展示部分结构，旋转视图不训练模型。示意向量及其分组不能证明模型已学会这些语义关系，要结合面板的数据说明理解。"
  ],
  "sampling": [
    "概率最大，是否意味着每次都会抽到它？",
    [
      "保留多个候选，连续采样并观察结果。",
      "只改变温度，对比概率柱与固定 logits。",
      "把 top-k 调到 1，再观察采样结果。"
    ],
    "多个候选时，按概率抽样仍有随机性。k=1 只剩一个候选；温度和截断调整选择分布，不改变模型权重，也不保证语义正确。"
  ],
  "pos-encoding": [
    "不同特征用相同的频率记录位置吗？",
    [
      "比较位置 0 的各维数值。",
      "选中不同维度对，观察曲线变化快慢。"
    ],
    "位置 0 的 sin/cos 对是 0 和 1；不同频率产生不同的位置刻度，表的数字与 token 的内容是两回事。"
  ],
  "rope-rotate": [
    "同时挪动两个位置，固定向量的点积会变吗？",
    [
      "保持维度对不变，对照下方两组间隔 Δ=3 的点积。",
      "把句子改成更长的句子（最多 8 字），观察第二组位置移动后读数是否相同。",
      "比较 Δ=1 到 4 的读数，再切换维度对。"
    ],
    "固定原始 q、k 时，位置的影响通过相对位移进入点积。真实 q、k 的内容同样影响分数，不能只凭距离决定关注程度。"
  ],
  "gqa-map": [
    "四个 query 共享两组 K/V，会只剩两个输出吗？",
    [
      "点“处理一个 token”，追踪 Q0、Q1 的共享组。",
      "数一数输出头数，再对照 K/V 组数与缓存比例。"
    ],
    "仍有四个 query 头的输出，减少的是 K/V 组数。这里 K/V 缓存元素数减半，不等于全模型内存与运行时间减半。"
  ],
  "swiglu-gate": [
    "门控是一个 0 到 1 的概率吗？",
    [
      "把门的输入 z 分别调到负值、0 和较大正值。",
      "对照 SiLU 的值与乘到内容分支后的输出。"
    ],
    "SiLU(z)=z×sigmoid(z)，可为负或大于 1。它是可学习的特征缩放，不是开关概率；乘法后还要经过输出投影。"
  ],
  "kv-cache": [
    "多生成一个 token，旧位置都要再算一次吗？",
    [
      "先观察处理整个前文的 prefill。",
      "逐步生成，比较新增 K/V 与已保留的缓存。"
    ],
    "在固定参数的因果推理中，旧前缀的各层 K/V 不受后续 token 影响，可复用。每层各有缓存，以额外内存减少重算。"
  ],
  "optimizer-race": [
    "同一学习率对每种优化器都合适吗？",
    [
      "重置后单步看一次不同方向的移动。",
      "换学习率后再重置比较，避免比较不同起点。"
    ],
    "不同优化器对梯度历史的使用不同。同一个人工曲面上的轨迹不能直接预测其他模型的速度排名，也不能说明相同学习率最公平。"
  ],
  "norm-playground": [
    "改动一个格子，会影响哪些归一化结果？",
    [
      "点击原始值的一格，观察 BN 同列与 LN 同行。",
      "再换一批，比较当前统计与运行统计。"
    ],
    "BN 跨样本统计同一特征，LN 跨特征统计当前样本。修改一个数时，两者受影响的范围不同。"
  ],
  "next-token": [
    "EOS 后面那个位置，两行 mask 为什么不一样？",
    [
      "逐个点击输入行，对照允许矩阵的每一行与目标 y[t] = x[t+1]。",
      "关掉再打开“补 PAD”，数一数 loss mask 与 attention mask 各自在哪变 0。",
      "点“看 <EOS> 的两条路径”，解释为什么 attention 仍为 1、损失却是 0。"
    ],
    "EOS 是真实输入，后续位置仍可读取它，所以 attention mask 为 1；它的目标已是 PAD（−100），不计损失。PAD 作为 key 整列被 attention 屏蔽，作为 target 被 loss 忽略——两条路径各管一件事。"
  ]
};

  var base = (function () {
    var src = (document.currentScript && document.currentScript.src) || '';
    var cut = src.lastIndexOf('/');
    return cut >= 0 ? src.slice(0, cut + 1) : '';
  })();

  var loading = {};   // id -> script injected
  var pending = {};   // id -> [container, ...] waiting for the script
  var mounted = {};   // container -> { id, dispose }

  function mountInto(id, container) {
    var impl = window.DLWidget._impl[id];
    if (!impl || container.dataset.mounted) { return; }
    container.dataset.mounted = '1';
    var frame = document.createElement('div');
    frame.className = 'widget-frame';
    container.appendChild(frame);
    var dispose = impl(frame) || null;
    mounted[container] = { id: id, dispose: dispose };
  }

  function loadWidget(id, container) {
    if (!REGISTRY[id]) { return; }
    if (window.DLWidget._impl[id]) { mountInto(id, container); return; }
    (pending[id] = pending[id] || []).push(container);
    if (loading[id]) { return; }
    loading[id] = true;
    var script = document.createElement('script');
    script.src = base + REGISTRY[id].file;
    script.onerror = function () {
      var note = document.createElement('p');
      note.className = 'wg-note';
      note.textContent = '组件加载失败：' + REGISTRY[id].file;
      container.appendChild(note);
    };
    document.head.appendChild(script);
  }

  function prepare(mount) {
    var id = mount.dataset.widget;
    if (!REGISTRY[id]) { return; }
    if (!mount.id) { mount.id = 'widget-' + id; }
    mount.classList.add('is-ready');
    var head = document.createElement('div');
    head.className = 'widget-head';
    head.appendChild(document.createTextNode('交互演示'));
    mount.appendChild(head);
    var guide = GUIDES[id];
    if (guide) {
      var box = document.createElement('div'); box.className = 'widget-guide';
      var question = document.createElement('p'); question.className = 'guide-question'; question.textContent = guide[0];
      var steps = document.createElement('ol');
      guide[1].forEach(function (text) { var li = document.createElement('li'); li.textContent = text; steps.appendChild(li); });
      var answer = document.createElement('details');
      var summary = document.createElement('summary'); summary.textContent = '操作后，展开核对解释';
      var reason = document.createElement('p'); reason.textContent = guide[2];
      answer.append(summary, reason); box.append(question, steps, answer); mount.appendChild(box);
    }
    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            io.disconnect();
            loadWidget(id, mount);
          }
        });
      }, { rootMargin: '600px 0px' });
      io.observe(mount);
    } else {
      loadWidget(id, mount);
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.widget-mount').forEach(prepare);
    // Deep link / QA hook: docs/<slug>/?widget=<id> jumps to that widget and
    // mounts it immediately. Uses a native hash jump (temporarily disabling
    // smooth scrolling) so it lands deterministically.
    var wanted = new URLSearchParams(window.location.search).get('widget');
    if (wanted && REGISTRY[wanted]) {
      var target = document.querySelector('.widget-mount[data-widget="' + wanted + '"]');
      if (target) {
        var parent = target.parentElement;
        while (parent) { if (parent.tagName === 'DETAILS') { parent.open = true; } parent = parent.parentElement; }
        var root = document.documentElement;
        var smooth = root.style.scrollBehavior;
        root.style.scrollBehavior = 'auto';
        window.location.hash = 'widget-' + wanted;
        root.style.scrollBehavior = smooth;
        loadWidget(wanted, target);
      }
    }
  });

  window.DLWidget = {
    _impl: {},
    register: function (id, mountFn) {
      this._impl[id] = mountFn;
      (pending[id] || []).forEach(function (container) { mountInto(id, container); });
      pending[id] = [];
    },
  };
})();
