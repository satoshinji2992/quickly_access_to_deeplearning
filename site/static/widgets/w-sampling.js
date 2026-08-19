/* sampling — 温度 / top-k / top-p 三个旋钮怎样改写「下一个 token」的抽签筒。
 * logits 固定（峰：睡/觉；次峰：叫/跑；长尾：的/是/在/快；谷：大/红/猫），
 * 管线固定：softmax(τ) → top-k → top-p（在 top-k 幸存者重归一后累计）→ 再重归一 → 抽样。
 */
(function () {
  'use strict';

  var TOK = ['的', '是', '猫', '在', '睡', '觉', '跑', '大', '红', '快', '好', '叫'];
  var N = TOK.length;   // 12
  var LOGITS = [-4.0, -3.4, -5.0, -2.8, 3.2, 2.4, 0.6, -5.2, -4.6, -1.5, -0.6, 1.1];

  var DEF = { tau: 1.0, k: 12, p: 0.90 };
  var TRACK = 112;      // 主图柱区高度 px
  var LTRACK = 50;      // logits 柱区高度 px
  var MONO = 'ui-monospace, SFMono-Regular, Consolas, monospace';

  var CSS = '' +
    '.smp-means{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:0 0 4px}' +
    '.smp-mean{border:1px solid rgba(7,19,33,.25);background:#f4f2ec;padding:7px 8px}' +
    '.smp-mean b{display:block;font:800 9.5px ' + MONO + ';letter-spacing:.07em;color:#071321;margin-bottom:3px;text-transform:uppercase}' +
    '.smp-mean i{font-style:normal;font-size:10.5px;color:#5b6670;line-height:1.55;display:block}' +
    '.smp-legend{display:flex;align-items:center;gap:4px;font:700 9px ' + MONO + ';color:#65717c}' +
    '.smp-chipb{display:inline-block;width:12px;height:9px;background:#0b63f3;border:1px solid #071321}' +
    '.smp-chipg{display:inline-block;width:12px;height:9px;border:1px solid rgba(7,19,33,.35);background:repeating-linear-gradient(45deg,#e8e4d6 0 3px,#d9d4c3 3px 6px);margin-left:6px}' +
    '.smp-chart{display:flex;gap:3px;border:1px solid #d8d3c4;background:#fffef9;padding:8px 6px 6px}' +
    '.smp-col{flex:1 1 0;min-width:0;display:flex;flex-direction:column;align-items:center}' +
    '.smp-val{height:14px;font:700 9px ' + MONO + ';color:#071321;white-space:nowrap;line-height:14px}' +
    '.smp-val .smp-x{color:#c0392b;font-weight:800;margin-right:1px}' +
    '.smp-track{width:100%;height:' + TRACK + 'px;display:flex;align-items:flex-end;justify-content:center}' +
    '.smp-bar{width:68%;max-width:28px;min-height:1px;border:1px solid #071321;background:#0b63f3;transition:height .18s ease,background-color .15s}' +
    '.smp-col.is-out .smp-bar{border-color:rgba(7,19,33,.35);background:repeating-linear-gradient(45deg,#e8e4d6 0 4px,#d9d4c3 4px 8px)}' +
    '.smp-tok{margin-top:4px;font:800 11px ' + MONO + ';color:#071321;height:15px;line-height:15px}' +
    '.smp-col.is-out .smp-val,.smp-col.is-out .smp-tok{color:#9aa4ad}' +
    '.smp-col.is-hit .smp-bar{background:#c8ff47;box-shadow:2px 2px 0 #071321}' +
    '.smp-col.is-hit .smp-tok{background:#071321;color:#c8ff47;padding:0 3px}' +
    '.smp-lchart .smp-track{height:' + LTRACK + 'px}' +
    '.smp-lchart .smp-val{height:12px;font-size:8.5px;color:#45515c;line-height:12px}' +
    '.smp-lchart .smp-bar{width:60%;max-width:20px;background:#263541;border-color:#263541}' +
    '.smp-lchart .smp-tok{display:none}' +
    '.smp-hist{display:flex;flex-wrap:wrap;gap:4px;min-height:24px;align-items:center}' +
    '.smp-chip{border:1px solid rgba(7,19,33,.3);background:#fffef9;padding:2px 6px;font:800 10.5px ' + MONO + ';color:#071321}' +
    '.smp-chip.is-new{background:#c8ff47;border-color:#071321}' +
    '.smp-empty{font-size:11px;color:#65717c}';

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined) { node.textContent = text; }
    return node;
  }

  /* 完整管线：返回 softmax(τ) 概率、截断后重归一的抽样分布、存活标记与淘汰原因。 */
  function pipeline(tau, k, p) {
    var i, z, m = -Infinity, es = [], sum = 0;
    for (i = 0; i < N; i += 1) {
      z = LOGITS[i] / tau;                       // 温度：先除再 softmax
      es.push(z);
      if (z > m) { m = z; }
    }
    var exps = [];
    for (i = 0; i < N; i += 1) {
      var e = Math.exp(es[i] - m);               // 减最大值：指数参数 ≤ 0，不溢出
      exps.push(e); sum += e;
    }
    var probs = [];
    for (i = 0; i < N; i += 1) { probs.push(exps[i] / sum); }

    var order = [];
    for (i = 0; i < N; i += 1) { order.push(i); }
    order.sort(function (a, b) { return probs[b] - probs[a] || a - b; });

    var alive = [], reason = [];
    for (i = 0; i < N; i += 1) { alive.push(true); reason.push(''); }

    // top-k：按概率只留前 k 个
    if (k < N) {
      for (i = k; i < N; i += 1) { alive[order[i]] = false; reason[order[i]] = 'k'; }
    }

    // 幸存者先重归一，top-p 在其上降序累计
    var s = 0;
    for (i = 0; i < N; i += 1) { if (alive[i]) { s += probs[i]; } }
    var cum = 0, keptOrder = [];
    if (p < 1) {
      for (i = 0; i < N; i += 1) {
        var idx = order[i];
        if (!alive[idx]) { continue; }
        if (cum >= p) { alive[idx] = false; reason[idx] = 'p'; continue; }
        cum += probs[idx] / s;
        keptOrder.push(idx);
      }
    } else {
      for (i = 0; i < N; i += 1) { if (alive[order[i]]) { keptOrder.push(order[i]); } }
    }

    // 最终重归一 → 抽样分布
    var fs = 0;
    for (i = 0; i < N; i += 1) { if (alive[i]) { fs += probs[i]; } }
    var finals = [];
    for (i = 0; i < N; i += 1) { finals.push(alive[i] ? probs[i] / fs : 0); }
    return { probs: probs, finals: finals, alive: alive, reason: reason, keptOrder: keptOrder };
  }

  function mount(container) {
    container.innerHTML =
      '<style>' + CSS + '</style>' +
      '<p class="wg-title">采样：logits 怎么变成「下一个 token」的抽签筒</p>' +
      '<p class="wg-sub">词表 12 个候选。模型最后一层输出的 logits 固定不变（下方暗色柱）；三个旋钮只改它变成概率之后的形状 —— 先 softmax(τ)，再依次过 top-k、top-p，最后重归一成抽样分布。</p>' +
      '<div class="smp-means">' +
        '<div class="smp-mean"><b>τ 温度 · 调锋利</b><i>logits 先除以 τ 再 softmax：τ 小分布更尖、τ 大被压平。不淘汰任何 token。</i></div>' +
        '<div class="smp-mean"><b>top-k · 定额淘汰</b><i>按概率排序只留前 k 个，其余直接出局；k=1 相当于贪心解码，永远挑最大。</i></div>' +
        '<div class="smp-mean"><b>top-p · 按量淘汰</b><i>从大到小累计概率，留下恰好凑满 p 的最小集合（核采样）；p=1.0 不淘汰。</i></div>' +
      '</div>' +
      '<div class="wg-label"><span>采样概率（截断后重归一）</span><span class="smp-legend"><i class="smp-chipb"></i>保留<i class="smp-chipg"></i>已淘汰</span></div>' +
      '<div class="smp-chart" data-role="chart"></div>' +
      '<p class="wg-note" data-role="cutnote"></p>' +
      '<div class="wg-label"><span>原始 logits（固定 · 滑杆不动它）</span><span>峰 3.2 · 谷 −5.2</span></div>' +
      '<div class="smp-chart smp-lchart" data-role="lchart"></div>' +
      '<div class="wg-block">' +
        '<div class="wg-label"><span>温度 τ</span><span data-role="tauv"></span></div>' +
        '<input class="wg-slider" type="range" data-role="tau" min="0.1" max="3" step="0.05" value="1">' +
        '<div class="wg-label"><span>top-k 保留数</span><span data-role="kval"></span></div>' +
        '<input class="wg-slider" type="range" data-role="k" min="1" max="12" step="1" value="12">' +
        '<div class="wg-label"><span>top-p 累计阈值</span><span data-role="pval"></span></div>' +
        '<input class="wg-slider" type="range" data-role="p" min="0.05" max="1" step="0.05" value="0.9">' +
      '</div>' +
      '<div class="wg-readout">' +
        '<div class="wg-stat"><span>保留候选</span><b data-role="skeep"></b></div>' +
        '<div class="wg-stat"><span>分布熵 H · 满=ln12≈2.48</span><b data-role="sent"></b></div>' +
        '<div class="wg-stat"><span>最高概率</span><b data-role="stop"></b></div>' +
        '<div class="wg-stat"><span>最近抽中</span><b data-role="sdraw">—</b></div>' +
      '</div>' +
      '<div class="wg-controls">' +
        '<button type="button" class="wg-button is-primary" data-role="draw">采一次</button>' +
        '<button type="button" class="wg-button" data-role="draw20">连采 ×20</button>' +
        '<button type="button" class="wg-button" data-role="reset">重置参数</button>' +
      '</div>' +
      '<div class="wg-label"><span>抽样记录</span><span data-role="hcount">已采 0 次 · 新在最左</span></div>' +
      '<div class="smp-hist" data-role="hist"><span class="smp-empty">点「采一次」按当前分布抽签 —— 多采几次，出现频率会贴上蓝柱的高度。</span></div>' +
      '<p class="wg-note">顺序固定：softmax(τ) → top-k → top-p（在 top-k 幸存者重归一后累计）→ 再重归一 → 抽样。softmax 用减最大值的稳定写法 p<sub>i</sub> = exp(z<sub>i</sub> − max z) / Σ<sub>j</sub> exp(z<sub>j</sub> − max z)：指数参数恒 ≤ 0 不会溢出，τ=0.1 时谷与峰相差 8.4 个 logit，最尾一项指数到 −84，e<sup>−84</sup> ≈ 0，被安全地压成 0。</p>';

    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };
    var state = { tau: DEF.tau, k: DEF.k, p: DEF.p };
    var last = null, draws = 0, history = [], flashTimer = null;

    /* 主图 12 根柱。 */
    var chart = q('chart'), cols = [], i;
    for (i = 0; i < N; i += 1) {
      var col = el('div', 'smp-col');
      var val = el('div', 'smp-val');
      var track = el('div', 'smp-track');
      var bar = el('div', 'smp-bar');
      track.appendChild(bar);
      col.appendChild(val); col.appendChild(track); col.appendChild(el('div', 'smp-tok', TOK[i]));
      chart.appendChild(col);
      cols.push({ col: col, val: val, bar: bar });
    }

    /* 次视图：原始 logits，挂载时画一次，永不随滑杆变。 */
    var lchart = q('lchart'), zMin = LOGITS[0], zMax = LOGITS[0];
    for (i = 1; i < N; i += 1) {
      if (LOGITS[i] < zMin) { zMin = LOGITS[i]; }
      if (LOGITS[i] > zMax) { zMax = LOGITS[i]; }
    }
    for (i = 0; i < N; i += 1) {
      var lcol = el('div', 'smp-col');
      lcol.title = '「' + TOK[i] + '」 logit = ' + LOGITS[i].toFixed(1);
      lcol.appendChild(el('div', 'smp-val', LOGITS[i].toFixed(1)));
      var lt = el('div', 'smp-track');
      var lb = el('div', 'smp-bar');
      lb.style.height = Math.max(2, Math.round((LOGITS[i] - zMin) / (zMax - zMin) * (LTRACK - 4))) + 'px';
      if (LOGITS[i] === zMax) { lb.style.background = '#0b63f3'; lb.style.borderColor = '#0b63f3'; }
      lt.appendChild(lb);
      lcol.appendChild(lt);
      lcol.appendChild(el('div', 'smp-tok', TOK[i]));
      lchart.appendChild(lcol);
    }

    function cutNote(r) {
      var parts = [];
      if (state.k < N) {
        var names = [];
        for (var j = 0; j < r.keptOrder.length; j += 1) {
          if (r.keptOrder[j] !== undefined) { names.push(TOK[r.keptOrder[j]]); }
        }
        parts.push('top-k=' + state.k + ' 留「' + names.join(' ') + '」，其余 ' + (N - state.k) + ' 个直接出局；');
      } else {
        parts.push('top-k=12 不淘汰；');
      }
      if (state.p < 1) {
        /* 与 pipeline 一致：累计在 top-k 幸存者（reason ≠ 'k'）重归一后进行，
           即 top-p 动手之前的集合。 */
        var s = 0, m;
        for (m = 0; m < N; m += 1) { if (r.reason[m] !== 'k') { s += r.probs[m]; } }
        var cum = 0, steps = [], nk = 0, np = 0;
        for (j = 0; j < r.keptOrder.length; j += 1) {
          var idx = r.keptOrder[j];
          cum += r.probs[idx] / s;
          steps.push(TOK[idx] + ' ' + cum.toFixed(2));
        }
        for (m = 0; m < N; m += 1) {
          if (!r.alive[m]) { if (r.reason[m] === 'k') { nk += 1; } else { np += 1; } }
        }
        parts.push('top-p=' + state.p.toFixed(2) + '：幸存者重归一后降序累计 ' + steps.join(' → ') +
          ' ≥ ' + state.p.toFixed(2) + ' 停 —— 保留 ' + r.keptOrder.length + ' 个重归一抽样；出局的 ' + (nk + np) + ' 个 = k 淘汰 ' + nk + ' + p 淘汰 ' + np + '。');
      } else {
        parts.push('top-p=1.00 不淘汰。');
      }
      return parts.join('');
    }

    function render() {
      var r = pipeline(state.tau, state.k, state.p);
      last = r;
      var maxD = 0, i2, d;
      for (i2 = 0; i2 < N; i2 += 1) {
        d = r.alive[i2] ? r.finals[i2] : r.probs[i2];
        if (d > maxD) { maxD = d; }
      }
      var top = 0;
      for (i2 = 0; i2 < N; i2 += 1) {
        var c = cols[i2], alive = r.alive[i2];
        d = alive ? r.finals[i2] : r.probs[i2];
        c.col.className = 'smp-col' + (alive ? '' : ' is-out');
        c.bar.style.height = Math.max(d > 0 ? 2 : 1, Math.round(d / maxD * TRACK)) + 'px';
        c.val.innerHTML = alive
          ? r.finals[i2].toFixed(2)
          : '<span class="smp-x">✕</span>' + r.probs[i2].toFixed(2);
        c.col.title = alive
          ? '「' + TOK[i2] + '」 采样概率 ' + r.finals[i2].toFixed(3)
          : '「' + TOK[i2] + '」 截断前 ' + r.probs[i2].toFixed(3) + ' · 被 top-' + r.reason[i2] + ' 淘汰';
        if (r.finals[i2] > r.finals[top]) { top = i2; }
      }

      var H = 0;
      for (i2 = 0; i2 < N; i2 += 1) {
        if (r.alive[i2]) { H -= r.finals[i2] * Math.log(r.finals[i2]); }
      }
      q('tauv').textContent = 'τ = ' + state.tau.toFixed(2);
      q('kval').textContent = 'k = ' + state.k + (state.k >= N ? '（不截）' : '');
      q('pval').textContent = 'p = ' + state.p.toFixed(2) + (state.p >= 1 ? '（不截）' : '');
      q('skeep').textContent = r.keptOrder.length + ' / ' + N;
      q('sent').textContent = H.toFixed(2);
      q('stop').textContent = '「' + TOK[top] + '」' + r.finals[top].toFixed(2);
      q('cutnote').textContent = cutNote(r);
    }

    function renderHist() {
      var hist = q('hist');
      hist.innerHTML = '';
      if (!history.length) {
        hist.appendChild(el('span', 'smp-empty', '点「采一次」按当前分布抽签 —— 多采几次，出现频率会贴上蓝柱的高度。'));
      } else {
        for (var j2 = 0; j2 < history.length; j2 += 1) {
          hist.appendChild(el('span', 'smp-chip' + (j2 === 0 ? ' is-new' : ''), TOK[history[j2]]));
        }
      }
      q('hcount').textContent = '已采 ' + draws + ' 次 · 新在最左';
    }

    function clearFlash() {
      if (flashTimer) { window.clearTimeout(flashTimer); flashTimer = null; }
      cols.forEach(function (c) { c.col.classList.remove('is-hit'); });
    }

    function sample() {
      var x = Math.random(), acc = 0, pick = -1, i3;
      for (i3 = 0; i3 < N; i3 += 1) {
        if (!last.alive[i3]) { continue; }
        acc += last.finals[i3];
        if (x < acc) { pick = i3; break; }
      }
      if (pick < 0) {                       // 浮点兜底：取最后一个幸存者
        for (i3 = N - 1; i3 >= 0; i3 -= 1) { if (last.alive[i3]) { pick = i3; break; } }
      }
      return pick;
    }

    function doDraw(count) {
      clearFlash();
      var pick = -1;
      for (var n = 0; n < count; n += 1) {
        pick = sample();
        history.unshift(pick);
        draws += 1;
      }
      if (history.length > 18) { history.length = 18; }
      cols[pick].col.classList.add('is-hit');
      flashTimer = window.setTimeout(clearFlash, 650);
      q('sdraw').textContent = '「' + TOK[pick] + '」' + last.finals[pick].toFixed(2);
      renderHist();
    }

    q('tau').addEventListener('input', function () { state.tau = parseFloat(this.value); render(); });
    q('k').addEventListener('input', function () { state.k = parseInt(this.value, 10); render(); });
    q('p').addEventListener('input', function () { state.p = parseFloat(this.value); render(); });
    q('draw').addEventListener('click', function () { doDraw(1); });
    q('draw20').addEventListener('click', function () { doDraw(20); });
    q('reset').addEventListener('click', function () {
      state.tau = DEF.tau; state.k = DEF.k; state.p = DEF.p;
      q('tau').value = String(DEF.tau); q('k').value = String(DEF.k); q('p').value = String(DEF.p);
      draws = 0; history = [];
      q('sdraw').textContent = '—';
      clearFlash();
      render();
      renderHist();
    });

    render();
    renderHist();
    return function dispose() { clearFlash(); };
  }

  window.DLWidget.register('sampling', mount);
})();
