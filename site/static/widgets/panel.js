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
    "拖动 a 会改变斜率，拖动 b 会平移直线；下方红线表示每个点的预测误差。选择只更新 b 或同时更新，再点一次更新，就能看到同一起点的两种变化。",
    "b 无法单独修正斜率。梯度给出当前位置的小步更新方向；学习率过大时，更新后的损失反而可能更高。这里使用面板列出的教学数据。"
  ],
  "mlp-boundary": [
    "图上的每条连接都对应一个权重。调整同一个权重后，整片平面会重新分类，可以看到哪些位置的分数跨过了分类边界。",
    "一个参数会参与许多点的共同计算。边界来自多个 ReLU 的分段函数组合，单个神经元并不对应一段固定的圆弧。"
  ],
  "backprop-steps": [
    "“下一步”沿着这个样本的计算展开，先得到预测和损失，再把梯度传回各个权重。更新前后的损失使用同一个样本计算。",
    "权重收到的梯度是输入乘以上游梯度，ReLU 的负输入处则不传回梯度。这个样本上的改进能核对更新过程；分类效果还需要看其他样本。"
  ],
  "image-batch": [
    "点击缩略图会切换图片，三个色彩面板显示同一张图的 R、G、B 通道。相同坐标在各通道中的数值，共同组成一个像素。",
    "NCHW 的四个轴依次是图片、通道、高、宽。换一张图片改变 N 轴的索引，换一个颜色通道改变 C 轴的索引。"
  ],
  "conv-sweep": [
    "移动卷积窗口时，右侧输出会标出当前格子。输入窗口跟着位置变化，参与相乘的那一组核权重保持相同。",
    "每个输出是局部窗口的加权和，同一卷积核在整张图上复用。面板中的核是为方便观察而指定的；训练时这些权重会随梯度更新。"
  ],
  "pool-sweep": [
    "同一个窗口可以切换为取最大值或平均值。反向视图显示每个输出收到的梯度怎样分配回输入。",
    "MaxPool 把梯度交给选中的最大值，AvgPool 则在窗口内均分；重叠窗口的贡献相加。池化会丢失信息，通常无法由输出还原整个窗口。"
  ],
  "residual-block": [
    "identity 模式保留输入本身，projection 模式先改变 shortcut 的尺寸。走到加法节点时，两条分支应当具有相同的 shape。",
    "下方简化梯度图只展示额外的传播路径。实际梯度仍取决于主分支导数和激活函数，分支贡献也可能相互抵消。"
  ],
  "im2col-grad": [
    "点击输出格子，会同时选中输入窗口和展开矩阵中的对应行。反向视图把这些窗口的梯度逐个放回原来的像素位置。",
    "一个像素可能被多个窗口使用，所以放回去时要累加。直接 reshape 无法合并这些重叠的贡献。"
  ],
  "qkv-flow": [
    "逐步切换可以看到 Q/K 点积、遮挡、Softmax 和对 V 的加权求和。点击分数矩阵的格子，会显示这一项来自哪两个向量。",
    "Softmax 每行沿 key 位置归一化，得到的是读取前文的权重。下一 token 的词表概率要到 LM head 之后才会得到。"
  ],
  "causal-mask": [
    "训练输入和目标错开一位。选中一个 query 位置，就能看到允许它读取的 key；切到推理视图后，未来 token 尚未生成。",
    "位置 t 可以读取 0 到 t，但不能读取 t+1。因此修改后面的 token，不应改变前面位置的输出。"
  ],
  "token-embed-3d": [
    "点击一个 token 可以对照原始八维向量和三维投影。旋转视图只改变观察角度，向量本身没有变化。",
    "三维投影舍弃了部分信息。面板标注了数据来源，示意分组不等同于训练后的语义关系，屏幕距离也不等同于原空间距离。"
  ],
  "sampling": [
    "温度和截断会改变这组固定 logits 对应的采样分布。连续抽样时，概率大的候选出现得更频繁，但每次结果未必相同。",
    "top-k 为 1 时只保留最大项，其余设置下仍可能保留多个候选。采样改变的是选择规则，模型参数没有更新。"
  ],
  "pos-encoding": [
    "位置 0 的 sin/cos 对分别为 0 和 1。切换维度对之后，曲线会以不同速度变化，数值表也会跟着更新。",
    "每对维度使用不同频率记录位置；这些值只由位置决定，与当前位置是什么 token 无关。"
  ],
  "rope-rotate": [
    "下方两组点积使用相同的原始 q、k，并保持位置间隔 Δ=3。改变句子长度会移动第二组位置，切换维度对则改变旋转频率。",
    "固定 q、k 时，同时平移两个位置不会改变旋转后的点积；位置影响由相对位移进入。真实注意力分数还取决于 q、k 的内容。"
  ],
  "gqa-map": [
    "“处理一个 token”会依次显示四个 query 头怎样使用两组 K/V。Q0、Q1 共享一组 K/V，但各自计算读取权重和输出。",
    "输出仍有四个头，减少的是 K/V 组数。在这里，K/V 缓存的元素数减半；全模型内存和耗时还包括其他计算。"
  ],
  "swiglu-gate": [
    "滑块改变门分支的输入 z。面板同时显示 SiLU(z)、内容分支和逐元素乘积，负输入和较大正输入尤其容易看出门值的范围。",
    "SiLU(z)=z×sigmoid(z)，可以为负，也可以大于 1。乘法调整各特征的幅度，之后还要经过输出投影。"
  ],
  "kv-cache": [
    "prefill 一次处理已有前文，随后每步只为新 token 计算并追加 K/V。图中每一层都维护自己的缓存。",
    "固定参数的因果推理中，旧位置不会读取未来 token，因此可以复用它们的 K/V。新 query 仍要与缓存中的 key 计算注意力。"
  ],
  "optimizer-race": [
    "几种优化器从同一点出发，面板显示它们在同一曲面上的更新轨迹。改变学习率后点击重置，可以比较不同设置。",
    "各方法使用梯度历史的方式不同，合适的学习率也不同。这张人工曲面能说明更新规则，不能给所有模型排出统一的快慢顺序。"
  ],
  "norm-playground": [
    "点击原始矩阵的一格，可以看到 BatchNorm 同列和 LayerNorm 同行的输出一起变化。“换一批”还会改变当前批次统计量。",
    "这里的 BatchNorm 跨样本统计同一特征，LayerNorm 跨特征统计当前样本。运行统计量记录历史批次，评估时不再由当前批次重新估计。"
  ],
  "next-token": [
    "输入行的每一格都对应一个 query。点击 EOS 或开关 PAD，可以对照 attention mask、目标行和 loss mask 在同一位置的不同取值。",
    "这个样本在 EOS 之后没有真实目标，所以该位置不计损失；EOS 本身仍是有效输入。PAD 作为 key 被注意力屏蔽，作为目标则由损失函数忽略。"
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
      var description = document.createElement('p'); description.textContent = guide[0];
      var answer = document.createElement('details');
      var summary = document.createElement('summary'); summary.textContent = '计算说明';
      var reason = document.createElement('p'); reason.textContent = guide[1];
      answer.append(summary, reason); box.append(description, answer); mount.appendChild(box);
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
