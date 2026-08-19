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
    { boxes: ['x'], text: '输入 x：<b>(1,16,32,32)</b>，同时进入主分支与 shortcut。' },
    { boxes: ['c1'], text: '主分支第一段 Conv3×3 → BN → ReLU，通道 16→16。' },
    { boxes: ['c2'], text: '主分支第二段 Conv3×3 → BN（无 ReLU），输出 F(x)，<b>shape 仍 (1,16,32,32)</b>。' },
    { boxes: ['sc'], text: 'shortcut：S(x)=x 原样直通，不引入参数。' },
    { boxes: ['add'], text: '⊕ 相加：F(x)+S(x)，两分支 shape 必须一致。' },
    { boxes: ['y'], text: '最后 ReLU：<b>y = ReLU(F(x)+S(x))</b>；映射接近恒等时只需 F(x)≈0。' },
  ];
  var BWD_STEPS = [
    { boxes: ['y'], text: 'dL/dy 先过 ReLU 掩码：前向为负处梯度为 0。' },
    { boxes: ['add'], text: '⊕：<b>梯度原样复制给两条支路</b>，dL/dF = dL/dS。' },
    { boxes: ['c2'], text: '主支路：dL/dF 依次过 BN、Conv 的 Jacobian。' },
    { boxes: ['sc'], text: 'shortcut：Jacobian 为 I，梯度<b>不经缩放直通回 x</b>。' },
    { boxes: ['x'], text: '合流：<b>dL/dx = dL/dy·(J_F + J_S)</b>；J_S=I 保证有不衰减路径。' },
  ];

  function mount(container) {
    container.innerHTML =
      '<style>' + CSS + '</style>' +
      '<p class="wg-title">残差块：两条分支与一条梯度直通路</p>' +
      '<p class="wg-sub">y = ReLU(F(x)+S(x))。步进看前向与反向。</p>' +
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
        ? 'shortcut：梯度只经 Conv1×1 一次缩放。'
        : 'shortcut：Jacobian 为 I，梯度<b>不经缩放直通回 x</b>。';
      BWD_STEPS[4].text = state.projection
        ? '合流：<b>dL/dx = dL/dy·(J_F + J_S)</b>，J_S 来自 Conv1×1。'
        : '合流：<b>dL/dx = dL/dy·(J_F + J_S)</b>；J_S=I 保证有不衰减路径。';
      FWD_STEPS[3].text = state.projection
        ? 'shortcut：16→32 且 s2 时，用 Conv1×1+BN 对齐 shape。'
        : 'shortcut：S(x)=x 原样直通，不引入参数。';
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
      q('bnote').textContent = 'σ=' + sigma.toFixed(2) + '、N=' + N + '：普通输入端梯度 ' + fmtSci(plain) + '，残差 ≥1。';
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
