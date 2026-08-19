/* residual-block — BasicBlock 两分支结构步进（前向合流 / 反向分流），
 * 加一个"深度 vs 梯度"简化数值实验：无残差时梯度随深度指数衰减，
 * 有残差时 shortcut 提供一条不衰减的直通路径。
 */
(function () {
  'use strict';

  var INK = '#071321', BLUE = '#0b63f3', RED = '#c0392b', GREEN = '#c8ff47', PAPER = '#f7f5ef';
  var MONO = 'ui-monospace,SFMono-Regular,Consolas,monospace';

  var CSS = '' +
    '.rs-flow{display:grid;grid-template-columns:auto 1fr auto;gap:10px 14px;align-items:center;margin:12px 0}' +
    '.rs-node{border:1.5px solid #071321;background:#fffef9;padding:8px 10px;text-align:center;' +
    'font:700 10.5px ' + MONO + ';color:#071321;line-height:1.45;min-width:64px;' +
    'transition:box-shadow .18s ease,background-color .18s ease}' +
    '.rs-node i{display:block;font-style:normal;font-size:8.5px;color:#65717c}' +
    '.rs-node.is-on{background:#071321;color:#fffef9;box-shadow:3px 3px 0 #0b63f3}' +
    '.rs-node.is-on i{color:#c8ff47}' +
    '.rs-node.is-dim{opacity:.38}' +
    '.rs-branch{display:flex;flex-direction:column;gap:10px}' +
    '.rs-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}' +
    '.rs-lab{font:800 9px ' + MONO + ';color:#0b63f3;flex:0 0 auto}' +
    '.rs-arrow{font:800 12px ' + MONO + ';color:#65717c}' +
    '.rs-arrow.bwd{color:' + RED + '}' +
    '.rs-add{width:46px;height:46px;border-radius:50%;border:2px solid #071321;display:grid;place-items:center;' +
    'font:800 20px ' + MONO + ';background:#fffef9;justify-self:center}' +
    '.rs-add.is-on{background:#071321;color:#c8ff47;box-shadow:3px 3px 0 #0b63f3}' +
    '.rs-out{display:flex;flex-direction:column;gap:8px;justify-items:center}' +
    '.rs-desc{border:1px solid rgba(7,19,33,.3);background:#f4f2ec;padding:9px 12px;margin:10px 0 0;' +
    'font:600 11px ' + MONO + ';color:#263541;line-height:1.7;min-height:44px}' +
    '.rs-desc b{color:' + BLUE + '}' +
    /* 面板 B：深度-梯度 */
    '.rs-bars{display:flex;flex-direction:column;gap:10px;margin:12px 0 4px}' +
    '.rs-bar{display:grid;grid-template-columns:120px 1fr 84px;gap:10px;align-items:center}' +
    '.rs-bar>span{font:700 10px ' + MONO + ';color:#45515c;text-align:right}' +
    '.rs-track{height:22px;border:1px solid rgba(7,19,33,.35);background:#fffef9;position:relative}' +
    '.rs-fill{height:100%;transition:width .3s ease}' +
    '.rs-fill.plain{background:' + RED + '}' +
    '.rs-fill.res{background:' + BLUE + '}' +
    '.rs-val{font:800 11px ' + MONO + ';color:#071321}' +
    '.rs-ticks{display:flex;justify-content:space-between;font:700 8px ' + MONO + ';color:#65717c;' +
    'margin:0 0 0 130px;padding-right:94px}';

  /* 面板 A 的步进描述：6 步前向 + 5 步反向。 */
  var FWD_STEPS = [
    { boxes: ['x'], text: '输入 x：<b>(1, 16, 32, 32)</b>。它同时进入主分支和 shortcut——两条路并行。' },
    { boxes: ['c1'], text: '主分支第一段 Conv3×3 → BN → ReLU：通道 16→16，得到中旬特征。' },
    { boxes: ['c2'], text: '主分支第二段 Conv3×3 → BN（无 ReLU）：输出 F(x)，<b>shape 仍 (1,16,32,32)</b>。' },
    { boxes: ['sc'], text: 'shortcut 计算 S(x)：identity 模式下 S(x)=x 原样直通（不引入任何参数）。' },
    { boxes: ['add'], text: '⊕ 相加：F(x)+S(x)。两条分支的 shape 必须完全一致，否则无法逐元素相加。' },
    { boxes: ['y'], text: '最后 ReLU：<b>y = ReLU(F(x)+S(x))</b>。若最合适的映射接近恒等，主分支只需把 F(x) 学到接近 0。' },
  ];
  var BWD_STEPS = [
    { boxes: ['y'], text: '反向开始：上游梯度 dL/dy 到达，先过 ReLU 掩码（前向为负的位置梯度为 0）。' },
    { boxes: ['add'], text: '梯度到达 ⊕：加法节点的梯度<b>原样复制给两条支路</b>——dL/dF = dL/dS = dL/d(和)。' },
    { boxes: ['c2'], text: '主支路：dL/dF 依次穿过 BN、Conv 的 Jacobian，每过一层都被缩放/混合一次。' },
    { boxes: ['sc'], text: 'shortcut：identity 的 Jacobian 是 I，梯度<b>不经任何缩放直通回 x</b>——这就是残差网络的梯度高速路。' },
    { boxes: ['x'], text: '合流：<b>dL/dx = dL/dy·(J_F + J_S)</b>。即使 J_F 很小，J_S=I 保证总有一条不衰减的路径。' },
  ];

  function mount(container) {
    container.innerHTML =
      '<style>' + CSS + '</style>' +
      '<p class="wg-title">残差块：两条分支与一条梯度直通路</p>' +
      '<p class="wg-sub">BasicBlock = 主分支学增量 F(x)，shortcut 提供 S(x)，相加后再 ReLU。步进看一遍前向合流与反向分流，再往下用简化模型看"越深梯度越小"的问题怎样被 shortcut 化解。</p>' +
      '<div class="wg-controls">' +
        '<button type="button" class="wg-button is-primary" data-role="mode-id">shortcut：identity（16→16）</button>' +
        '<button type="button" class="wg-button" data-role="mode-pr">projection（16→32, s2）</button>' +
      '</div>' +
      '<div class="rs-flow">' +
        '<div class="rs-node" data-box="x">x<i>(1,16,32,32)</i></div>' +
        '<div class="rs-branch">' +
          '<div class="rs-row"><span class="rs-lab">主分支</span>' +
            '<div class="rs-node" data-box="c1">Conv3×3·BN·ReLU<i>16→16</i></div>' +
            '<span class="rs-arrow">→</span>' +
            '<div class="rs-node" data-box="c2">Conv3×3·BN<i>F(x)</i></div>' +
          '</div>' +
          '<div class="rs-row"><span class="rs-lab">shortcut</span>' +
            '<div class="rs-node" data-box="sc" data-role="scnode">S(x)=x<i>identity · 0 参数</i></div>' +
          '</div>' +
        '</div>' +
        '<div class="rs-add" data-box="add">⊕</div>' +
      '</div>' +
      '<div class="rs-flow" style="grid-template-columns:auto auto auto auto;justify-content:start;gap:10px">' +
        '<span class="rs-arrow">⊕ →</span>' +
        '<div class="rs-node" data-box="y">ReLU → y<i data-role="yshape">(1,16,32,32)</i></div>' +
        '<span class="rs-arrow" style="margin-left:8px">反向：</span>' +
        '<span class="rs-arrow bwd">dL/dy ⟶ ⊕ ⟶ (dL/dF ‖ dL/dS) ⟶ dL/dx</span>' +
      '</div>' +
      '<div class="wg-controls">' +
        '<button type="button" class="wg-button" data-role="prev">← 上一步</button>' +
        '<button type="button" class="wg-button is-primary" data-role="next">下一步 →</button>' +
        '<button type="button" class="wg-button" data-role="auto">自动播放</button>' +
        '<button type="button" class="wg-button" data-role="reset">重置</button>' +
      '</div>' +
      '<div class="rs-desc" data-role="desc"></div>' +

      '<div class="wg-label" style="margin-top:22px"><span>深度-梯度实验（简化模型）</span><span>条形为对数尺度</span></div>' +
      '<p class="wg-note">把每一层对梯度范数的"保留率"记为 σ（卷积混合会缩小它）。普通深网把 N 层连乘，到达输入的梯度是 σᴺ；有 shortcut 时，加法节点让梯度多一条不衰减的直通路，最底层至少收到 1 份。</p>' +
      '<div class="wg-readout">' +
        '<div class="wg-stat"><span>深度 N（层）</span><b data-role="nv">12</b></div>' +
        '<div class="wg-stat"><span>每层保留率 σ</span><b data-role="sv">0.60</b></div>' +
      '</div>' +
      '<input class="wg-slider" type="range" data-role="n" min="2" max="24" step="1" value="12">' +
      '<input class="wg-slider" type="range" data-role="s" min="20" max="95" step="5" value="60">' +
      '<div class="rs-bars">' +
        '<div class="rs-bar"><span>普通 N 层 σᴺ</span><div class="rs-track"><div class="rs-fill plain" data-role="bplain"></div></div><b class="rs-val" data-role="vplain"></b></div>' +
        '<div class="rs-bar"><span>残差 · 直通路 ≥1</span><div class="rs-track"><div class="rs-fill res" data-role="bres"></div></div><b class="rs-val" data-role="vres"></b></div>' +
      '</div>' +
      '<div class="rs-ticks"><span>1e-9</span><span>1e-6</span><span>1e-3</span><span>1</span></div>' +
      '<p class="wg-note" data-role="bnote"></p>';

    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };
    var boxes = {};
    container.querySelectorAll('[data-box]').forEach(function (el) { boxes[el.getAttribute('data-box')] = el; });

    var state = { step: 4, projection: false, timer: null };   /* 初始停在 ⊕ 相加这一步，截图有信息量 */
    var TOTAL = FWD_STEPS.length + BWD_STEPS.length;

    function allSteps() { return FWD_STEPS.concat(BWD_STEPS); }

    function paintStep() {
      var steps = allSteps();
      var idx = Math.max(0, Math.min(TOTAL - 1, state.step));
      var cur = steps[idx];
      var activeSet = {};
      cur.boxes.forEach(function (b) { activeSet[b] = true; });
      Object.keys(boxes).forEach(function (key) {
        boxes[key].classList.toggle('is-on', !!activeSet[key]);
        boxes[key].classList.toggle('is-dim', !activeSet[key]);
      });
      var phase = idx < FWD_STEPS.length ? '前向' : '反向';
      q('desc').innerHTML = '<b>第 ' + (idx + 1) + '/' + TOTAL + ' 步 · ' + phase + '</b>　' + cur.text;
    }

    function paintMode() {
      var idBtn = q('mode-id'), prBtn = q('mode-pr');
      idBtn.classList.toggle('is-primary', !state.projection);
      prBtn.classList.toggle('is-primary', state.projection);
      var sc = q('scnode');
      if (state.projection) {
        sc.innerHTML = 'Conv1×1·BN<i>S(x)=W·x · s2 对齐</i>';
      } else {
        sc.innerHTML = 'S(x)=x<i>identity · 0 参数</i>';
      }
      q('yshape').textContent = state.projection ? '(1,32,16,16)' : '(1,16,32,32)';
      BWD_STEPS[3].text = state.projection
        ? 'shortcut：projection 的 Conv1×1 也是浅层线性映射，梯度只经过一次转置缩放，仍远比主分支直接。'
        : 'shortcut：identity 的 Jacobian 是 I，梯度<b>不经任何缩放直通回 x</b>——这就是残差网络的梯度高速路。';
      BWD_STEPS[4].text = state.projection
        ? '合流：<b>dL/dx = dL/dy·(J_F + J_S)</b>，J_S 来自 Conv1×1 的转置。两条路相加，衰减不再是唯一命运。'
        : '合流：<b>dL/dx = dL/dy·(J_F + J_S)</b>。即使 J_F 很小，J_S=I 保证总有一条不衰减的路径。';
      FWD_STEPS[3].text = state.projection
        ? 'shortcut 计算 S(x)：通道 16→32 且 stride 2 时 x 无法直接相加，用 Conv1×1+BN 对齐 shape。'
        : 'shortcut 计算 S(x)：identity 模式下 S(x)=x 原样直通（不引入任何参数）。';
      paintStep();
    }

    /* ---- 面板 B：对数尺度条形 ---- */
    var LOG_MIN = -9, LOG_MAX = 0;
    function fmtSci(v) {
      if (v >= 0.001) { return v.toFixed(3); }
      var e = Math.floor(Math.log10(v));
      var m = v / Math.pow(10, e);
      return m.toFixed(1) + 'e' + e;
    }
    function paintBars() {
      var N = parseInt(q('n').value, 10);
      var sigma = parseInt(q('s').value, 10) / 100;
      q('nv').textContent = String(N);
      q('sv').textContent = sigma.toFixed(2);
      var plain = Math.pow(sigma, N);
      var width = function (v) {
        var L = Math.max(LOG_MIN, Math.min(LOG_MAX, Math.log10(Math.max(v, 1e-12))));
        return Math.max(1.5, (L - LOG_MIN) / (LOG_MAX - LOG_MIN) * 100);
      };
      q('bplain').style.width = width(plain) + '%';
      q('bres').style.width = width(1) + '%';
      q('vplain').textContent = fmtSci(plain);
      q('vres').textContent = '≥ 1';
      q('bnote').textContent = 'σ=' + sigma.toFixed(2) + '、N=' + N + '：普通网络的输入端梯度只剩 ' +
        fmtSci(plain) + '（连乘 ' + N + ' 次）；残差网络无论多深，shortcut 都把完整的一份梯度送回输入。' +
        '真实网络里 σ 不是常数，这里只演示连乘衰减与直通路保底的对比。';
    }

    /* ---- 事件 ---- */
    q('next').addEventListener('click', function () { state.step += 1; if (state.step >= TOTAL) { state.step = 0; } paintStep(); });
    q('prev').addEventListener('click', function () { state.step -= 1; if (state.step < 0) { state.step = TOTAL - 1; } paintStep(); });
    q('reset').addEventListener('click', function () { state.step = 4; paintStep(); });
    q('auto').addEventListener('click', function () {
      if (state.timer) {
        window.clearInterval(state.timer); state.timer = null;
        q('auto').classList.remove('is-primary');
        return;
      }
      q('auto').classList.add('is-primary');
      state.timer = window.setInterval(function () {
        state.step = (state.step + 1) % TOTAL;
        paintStep();
      }, 1400);
    });
    q('mode-id').addEventListener('click', function () { state.projection = false; paintMode(); });
    q('mode-pr').addEventListener('click', function () { state.projection = true; paintMode(); });
    q('n').addEventListener('input', paintBars);
    q('s').addEventListener('input', paintBars);

    paintMode();
    paintBars();

    return function dispose() {
      if (state.timer) { window.clearInterval(state.timer); state.timer = null; }
    };
  }

  window.DLWidget.register('residual-block', mount);
})();
