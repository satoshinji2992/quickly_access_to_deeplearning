/* pos-encoding — 正弦位置编码：把位置 m 写成多尺度 sin/cos 向量（绝对位置）。
 * D=16（8 个维度对）、base=10000。RoPE 的旋转演示见 w-rope-rotate.js。
 */
(function () {
  'use strict';

  var INK = '#071321', BLUE = '#0b63f3', RED = '#c0392b', PAPER = '#fffef9';
  var MONO = 'ui-monospace,SFMono-Regular,Consolas,monospace';
  var D = 16, NP = 8, MAXT = 8, TAU = Math.PI * 2;

  function omega(i) { return Math.pow(10000, -2 * i / D); }          // ω_i，i = 0..7
  function peVal(m, d) { var w = omega(Math.floor(d / 2)); return d % 2 === 0 ? Math.sin(m * w) : Math.cos(m * w); }

  function hexRgb(h) { return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
  var CBLUE = hexRgb(BLUE), CRED = hexRgb(RED), CPAPE = hexRgb(PAPER);
  function divColor(v) {   // 发散：蓝（负）— 纸白（0）— 红（正），|v| 越大越深
    var t = Math.max(0, Math.min(1, Math.abs(v)));
    var c = v >= 0 ? CRED : CBLUE, out = [];
    for (var k = 0; k < 3; k++) { out.push(Math.round(CPAPE[k] + (c[k] - CPAPE[k]) * t)); }
    return 'rgb(' + out.join(',') + ')';
  }

  function fmtW(w) { return w >= 0.01 ? w.toFixed(2) : w.toExponential(1); }
  function fmtPeriod(w) { var p = TAU / w; return p >= 100 ? String(Math.round(p)) : p.toFixed(1); }
  function showCh(ch) { return /^\s$/.test(ch) ? '␣' : ch; }

  var CSS = '' +
    '.pe-row{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start;margin:12px 0}' +
    '.pe-in{width:100%;box-sizing:border-box;border:1px solid #071321;background:#fffef9;padding:8px 10px;font:600 14px/1.5 ' + MONO + ';color:#071321}' +
    '.pe-in:focus{outline:none;box-shadow:3px 3px 0 rgba(11,99,243,.55)}' +
    '.pe-col-hm{flex:0 0 324px;max-width:100%;min-width:262px}' +
    '.pe-col-cur{flex:1 1 220px;min-width:220px}' +
    '.pe-hm{display:grid;grid-template-columns:40px repeat(16,minmax(0,1fr));gap:3px;user-select:none}' +
    '.pe-corner{display:grid;place-items:center;border:1px solid rgba(7,19,33,.25);background:#f4f2ec;font:700 8px ' + MONO + ';color:#65717c}' +
    '.pe-ph{grid-column:span 2;display:flex;flex-direction:column;align-items:center;justify-content:center;line-height:1.3;border:1px solid rgba(7,19,33,.3);background:#f4f2ec;font:700 8px ' + MONO + ';color:#45515c;cursor:pointer;padding:2px 0}' +
    '.pe-ph b{font:800 9px ' + MONO + ';color:#071321}' +
    '.pe-ph i{font-style:normal;font-size:7.5px}' +
    '.pe-ph:hover{border-color:#0b63f3}' +
    '.pe-ph.is-sel{background:#071321;border-color:#071321}' +
    '.pe-ph.is-sel b{color:#c8ff47}' +
    '.pe-ph.is-sel i{color:rgba(255,254,249,.72)}' +
    '.pe-rh{display:flex;flex-direction:column;justify-content:center;gap:1px;border:1px solid rgba(7,19,33,.3);background:#f4f2ec;padding:2px 5px;line-height:1.2}' +
    '.pe-rh b{font:800 12px ' + MONO + ';color:#071321}' +
    '.pe-rh span{font:700 8px ' + MONO + ';color:#65717c}' +
    '.pe-cell{height:24px;border:1px solid rgba(7,19,33,.22);background:#fffef9;cursor:pointer}' +
    '.pe-cell:hover{border-color:#071321}' +
    '.pe-cell.is-sel{border-color:#071321;box-shadow:0 0 0 2px #c8ff47;position:relative;z-index:2}';

  function mount(container) {
    container.innerHTML =
      '<style>' + CSS + '</style>' +
      '<p class="wg-title">正弦位置编码：多尺度频率</p>' +
      '<p class="wg-sub">句子按字符切成 T 个 token，每个 token 的位置 m 被写成 D 维 sin/cos 向量，直接加到 embedding 上（绝对位置）。取 D=16（8 个维度对）、base=10000。</p>' +
      '<div class="wg-label"><span>句子（按字符切分，最多 8 字）</span><span data-role="cnt"></span></div>' +
      '<input class="pe-in" data-role="sent" value="小猫在睡觉">' +
      '<p class="wg-note" data-role="snote"></p>' +
      '<div class="wg-label"><span>PE 热图 shape (T,16) · 蓝 = 负 · 白 = 0 · 红 = 正</span><span>← 低维变化快 · 高维变化慢 →</span></div>' +
      '<div class="pe-row">' +
        '<div class="pe-col-hm">' +
          '<div class="pe-hm" data-role="hm"></div>' +
          '<p class="wg-note">行 = token 位置 m，列 = 维度 d。d0/d1 共用角频率 ω0，d2/d3 共用 ω1……列头同时标出每个维度对的 ω。点击任一列选中该维度对，右侧画出它的 sin/cos。</p>' +
        '</div>' +
        '<div class="pe-col-cur">' +
          '<div class="wg-label"><span data-role="curtitle"></span><span>纵轴 [−1,1]</span></div>' +
          '<canvas class="wg-canvas" data-role="cur" height="180"></canvas>' +
          '<div class="wg-readout">' +
            '<div class="wg-stat"><span>角频率 ω</span><b data-role="om"></b></div>' +
            '<div class="wg-stat"><span>周期 2π/ω（位置数）</span><b data-role="per"></b></div>' +
          '</div>' +
          '<p class="wg-note" data-role="curnote"></p>' +
        '</div>' +
      '</div>';

    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };
    var cvCur = q('cur'), hm = q('hm');
    var state = { pair: 0, tokens: [] };
    var timers = [];
    function later(fn, ms) { timers.push(window.setTimeout(fn, ms)); }

    // HiDPI：显式钉住 CSS 尺寸，画布按 devicePixelRatio 放大（逻辑尺寸恒为常量）。
    function dpr(canvas, logicalHeight) {
      var cssWidth = canvas.parentElement ? canvas.parentElement.clientWidth : 0;
      cssWidth = Math.max(200, Math.min(cssWidth || 400, 640));
      var ratio = window.devicePixelRatio || 1;
      canvas.style.width = cssWidth + 'px';
      canvas.style.height = logicalHeight + 'px';
      canvas.width = Math.round(cssWidth * ratio);
      canvas.height = Math.round(logicalHeight * ratio);
      var ctx = canvas.getContext('2d');
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return { ctx: ctx, w: cssWidth, h: logicalHeight };
    }

    function el(tag, cls, text) {
      var node = document.createElement(tag);
      if (cls) { node.className = cls; }
      if (text !== undefined) { node.textContent = text; }
      return node;
    }

    function tokenize() {
      var raw = q('sent').value;
      var all = Array.from(raw);
      var note;
      if (!all.length) {
        state.tokens = ['□'];
        note = '输入为空：暂用占位 token □ 演示（T=1）。';
      } else {
        state.tokens = all.slice(0, MAXT);
        note = all.length > MAXT
          ? '句子过长：切分出 ' + all.length + ' 个字符，已截断为前 ' + MAXT + ' 个。'
          : 'T=' + state.tokens.length + '：每个字符一个位置 m=0..' + (state.tokens.length - 1) + '。';
      }
      q('cnt').textContent = state.tokens.length + ' / ' + MAXT + ' 字符';
      q('snote').textContent = note;
    }

    function buildHeatmap() {
      var T = state.tokens.length;
      hm.innerHTML = '';
      hm.appendChild(el('div', 'pe-corner', 'm ╲ i'));
      for (var i = 0; i < NP; i++) {
        var sel = i === state.pair;
        var ph = el('div', 'pe-ph' + (sel ? ' is-sel' : ''));
        ph.appendChild(el('b', '', 'i=' + i));
        ph.appendChild(el('i', '', 'ω ' + fmtW(omega(i))));
        ph.title = '维度对 i=' + i + '（列 d' + (2 * i) + '、d' + (2 * i + 1) + '）：角频率 ω=' + fmtW(omega(i)) +
          '，周期 2π/ω ≈ ' + fmtPeriod(omega(i)) + ' 个位置。点击选中。';
        if (!sel) { ph.style.background = 'rgba(11,99,243,' + (0.26 - i * 0.031).toFixed(3) + ')'; }
        ph.addEventListener('click', function (idx) {
          return function () { state.pair = idx; render(); };
        }(i));
        hm.appendChild(ph);
      }
      for (var m = 0; m < T; m++) {
        var rh = el('div', 'pe-rh');
        rh.appendChild(el('b', '', showCh(state.tokens[m])));
        rh.appendChild(el('span', '', 'm=' + m));
        rh.title = 'token「' + state.tokens[m] + '」· 位置 m=' + m;
        hm.appendChild(rh);
        for (var d = 0; d < D; d++) {
          (function (mm, dd) {
            var v = peVal(mm, dd), pi = Math.floor(dd / 2);
            var cell = el('div', 'pe-cell' + (pi === state.pair ? ' is-sel' : ''));
            cell.style.background = divColor(v);
            cell.title = 'PE[' + mm + ', d' + dd + '] = ' + (dd % 2 ? 'cos' : 'sin') + '(' + mm + '·ω' + pi + ') = ' + v.toFixed(3) +
              ' · 点击选中维度对 i=' + pi;
            cell.addEventListener('click', function () { state.pair = pi; render(); });
            hm.appendChild(cell);
          }(m, d));
        }
      }
    }

    function drawCurves() {
      var T = state.tokens.length, w = omega(state.pair);
      var H = Math.max(160, 26 + T * 27);
      var c = dpr(cvCur, H), ctx = c.ctx;
      var L = 20, R = 8, TP = 34, B = 32;
      var pw = c.w - L - R, ph = c.h - TP - B;
      var X = function (m) { return L + (m + 0.5) / T * pw; };
      var Y = function (y) { return TP + (1 - y) / 2 * ph; };
      ctx.clearRect(0, 0, c.w, c.h);

      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(7,19,33,.12)';
      ctx.setLineDash([3, 3]);
      [-1, 1].forEach(function (y) {
        ctx.beginPath(); ctx.moveTo(L, Y(y)); ctx.lineTo(L + pw, Y(y)); ctx.stroke();
      });
      ctx.setLineDash([]);
      ctx.strokeStyle = 'rgba(7,19,33,.10)';
      for (var g = 0; g < T; g++) {
        ctx.beginPath(); ctx.moveTo(X(g), TP); ctx.lineTo(X(g), TP + ph); ctx.stroke();
      }
      ctx.strokeStyle = 'rgba(7,19,33,.30)';
      ctx.beginPath(); ctx.moveTo(L, Y(0)); ctx.lineTo(L + pw, Y(0)); ctx.stroke();
      ctx.fillStyle = '#65717c'; ctx.font = '9px ' + MONO; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText('+1', L - 3, Y(1)); ctx.fillText('0', L - 3, Y(0)); ctx.fillText('−1', L - 3, Y(-1));

      function curve(fn, style, dash) {
        ctx.strokeStyle = style; ctx.lineWidth = 1.8; ctx.setLineDash(dash || []);
        ctx.beginPath();
        var steps = Math.max(80, Math.round(pw));
        for (var s = 0; s <= steps; s++) {
          var m = -0.5 + (s / steps) * T;
          var x = X(m), y = Y(fn(m * w));
          if (s) { ctx.lineTo(x, y); } else { ctx.moveTo(x, y); }
        }
        ctx.stroke(); ctx.setLineDash([]);
      }
      curve(function (t) { return Math.cos(t); }, 'rgba(7,19,33,.75)', [6, 4]);
      curve(function (t) { return Math.sin(t); }, BLUE, []);

      var m2, t2;
      for (m2 = 0; m2 < T; m2++) {
        t2 = m2 * w;
        ctx.beginPath(); ctx.arc(X(m2), Y(Math.cos(t2)), 3.2, 0, TAU);
        ctx.fillStyle = PAPER; ctx.fill();
        ctx.strokeStyle = INK; ctx.lineWidth = 1.4; ctx.stroke();
        ctx.beginPath(); ctx.arc(X(m2), Y(Math.sin(t2)), 3.6, 0, TAU);
        ctx.fillStyle = BLUE; ctx.fill();
      }
      for (m2 = 0; m2 < T; m2++) {
        ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
        ctx.fillStyle = INK; ctx.font = '800 11px ' + MONO;
        ctx.fillText(showCh(state.tokens[m2]), X(m2), c.h - 16);
        ctx.fillStyle = '#65717c'; ctx.font = '8px ' + MONO;
        ctx.fillText('m=' + m2, X(m2), c.h - 5);
      }

      // 顶部直接图例（色样 + 等宽文字，避免只靠颜色区分）
      var lx = L + 2, ly1 = 12, ly2 = 25;
      ctx.strokeStyle = BLUE; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(lx, ly1); ctx.lineTo(lx + 20, ly1); ctx.stroke();
      ctx.fillStyle = INK; ctx.font = '700 9px ' + MONO; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText('sin(m·ω)：偶数维 d' + (2 * state.pair), lx + 25, ly1);
      ctx.strokeStyle = 'rgba(7,19,33,.75)'; ctx.lineWidth = 1.8; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(lx, ly2); ctx.lineTo(lx + 20, ly2); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillText('cos(m·ω)：奇数维 d' + (2 * state.pair + 1), lx + 25, ly2);
    }

    function render() {
      buildHeatmap();
      var w = omega(state.pair);
      q('curtitle').textContent = '维度对 i=' + state.pair + ' · d' + (2 * state.pair) + '/d' + (2 * state.pair + 1);
      q('om').textContent = fmtW(w);
      q('per').textContent = fmtPeriod(w);
      q('curnote').textContent = 'ω' + state.pair + ' = ' + fmtW(w) + '：走满一圈要 ' + fmtPeriod(w) +
        ' 个位置。低维（i 小）几个位置就变一圈，高维成百上千个位置才变一圈——每个 token 的编码由这些不同频率叠加，唯一标识位置 m。';
      drawCurves();
    }

    q('sent').addEventListener('input', function () { tokenize(); render(); });

    var onResize = function () { render(); };
    window.addEventListener('resize', onResize);

    tokenize();
    render();
    later(function () { render(); }, 60);   // 字体/布局就绪后再画一帧，防首轮宽度偏差

    return function dispose() {
      timers.forEach(function (t) { window.clearTimeout(t); });
      timers = [];
      window.removeEventListener('resize', onResize);
    };
  }

  window.DLWidget.register('pos-encoding', mount);
})();
