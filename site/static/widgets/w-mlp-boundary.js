/* mlp-boundary — 手调 2-h-2 MLP 的权重，看决策边界如何在圆形数据上从直线弯成圆。 */
(function () {
  'use strict';

  var INK = '#071321', BLUE = '#0b63f3', RED = '#c0392b', PAPER = '#fffef9', BEIGE = '#f2e9d6';
  var DOM = 1.6;           // 画布范围 [-1.6, 1.6]^2
  var STEP = 0.3;          // 每次点击的权重增量

  /* ---------- 数据：单位圆内=1、圆外=0，固定种子保证可复现 ---------- */
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function genData() {
    var rand = mulberry32(20260819);
    var pts = [];
    for (var i = 0; i < 40; i++) {
      var ang = rand() * Math.PI * 2;
      var rr = 0.96 * Math.sqrt(rand());          // 圆内留一点边距
      pts.push({ x: rr * Math.cos(ang), y: rr * Math.sin(ang), c: 1 });
    }
    for (var j = 0; j < 40; j++) {
      var ang2 = rand() * Math.PI * 2;
      var lo = 1.10, hi = 1.55;                    // 1.10~1.55 的环带
      var rr2 = Math.sqrt(lo * lo + rand() * (hi * hi - lo * lo));
      pts.push({ x: rr2 * Math.cos(ang2), y: rr2 * Math.sin(ang2), c: 0 });
    }
    return pts;
  }

  /* ---------- 前向：2 -> h(ReLU) -> 2，纯 JS ---------- */
  function forward(m, x, y) {
    var hout = new Array(m.h);
    for (var j = 0; j < m.h; j++) {
      hout[j] = Math.max(0, m.w1[j][0] * x + m.w1[j][1] * y + m.b1[j]);
    }
    var o0 = m.b2[0], o1 = m.b2[1];
    for (var k = 0; k < m.h; k++) {
      o0 += m.w2[0][k] * hout[k];
      o1 += m.w2[1][k] * hout[k];
    }
    return { h: hout, d: o1 - o0 };               // d>0 => 预测圆内(1)
  }

  function cloneModel(m) {
    return {
      h: m.h,
      w1: m.w1.map(function (r) { return r.slice(); }),
      b1: m.b1.slice(),
      w2: [m.w2[0].slice(), m.w2[1].slice()],
      b2: m.b2.slice(),
    };
  }

  /* ---------- 两种预设 ---------- */
  function presetLine(h) {
    // 权重都很小且方向几乎一致 => 边界近似一条直线
    var m = { h: h, w1: [], b1: [], w2: [[], []], b2: [0, 0.15] };
    for (var j = 0; j < h; j++) {
      var s = (j % 2 === 0) ? 1 : -1;
      m.w1[j] = [s * 0.55, s * 0.38];
      m.b1[j] = -0.08 * s;
      m.w2[0][j] = (j % 2 === 0) ? 0.9 : -0.9;
      m.w2[1][j] = -m.w2[0][j];
    }
    return m;
  }

  function presetCircle(h) {
    // h 个方向上的半平面：ReLU(u·x - r) 只在圆外那侧激活，
    // 输出层把激活累加给"圆外"、减给"圆内" => 围出近似圆的 h 边形
    var m = { h: h, w1: [], b1: [], w2: [[], []], b2: [0, 0.35] };
    var A = 2.2, r = 1.03;
    for (var j = 0; j < h; j++) {
      var th = Math.PI * 2 * (j + 0.5) / h;
      m.w1[j] = [A * Math.cos(th), A * Math.sin(th)];
      m.b1[j] = -A * r;
      m.w2[0][j] = 1.4;
      m.w2[1][j] = -1.4;
    }
    return m;
  }

  function randomModel(h) {
    var m = { h: h, w1: [], b1: [], w2: [[], []], b2: [] };
    for (var j = 0; j < h; j++) {
      m.w1[j] = [(Math.random() * 2 - 1) * 1.2, (Math.random() * 2 - 1) * 1.2];
      m.b1[j] = (Math.random() * 2 - 1) * 0.8;
    }
    for (var k = 0; k < h; k++) {
      m.w2[0][k] = (Math.random() * 2 - 1) * 1.3;
      m.w2[1][k] = (Math.random() * 2 - 1) * 1.3;
    }
    m.b2 = [(Math.random() * 2 - 1) * 0.5, (Math.random() * 2 - 1) * 0.5];
    return m;
  }

  function clampW(v) { return Math.max(-4, Math.min(4, Math.round(v * 10) / 10)); }

  function fmt(v) { return (v < 0 ? '−' : '+') + Math.abs(v).toFixed(1); }
  function fmtB(v) { return (v < 0 ? '−' : '+') + Math.abs(v).toFixed(1); }

  /* ---------- SVG 工具 ---------- */
  var NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, parent) {
    var node = document.createElementNS(NS, tag);
    for (var k in attrs) { node.setAttribute(k, attrs[k]); }
    if (parent) { parent.appendChild(node); }
    return node;
  }

  function mount(container) {
    container.innerHTML =
      '<p class="wg-title">MLP 决策边界：手调一个 2-h-2 网络</p>' +
      '<p class="wg-sub">数据只有两个特征 (x₁, x₂)：单位圆内为蓝色类、圆外为米色类。下面的网络图里，每条边就是一个权重——点击边 +0.3、Shift+点击 −0.3，看边界跟着弯折。</p>' +
      '<div class="wg-label"><span>隐藏宽度 h（神经元个数）</span><span data-role="hval">2-4-2</span></div>' +
      '<div class="wg-controls">' +
        '<button type="button" class="wg-button" data-role="h2">h=2</button>' +
        '<button type="button" class="wg-button is-primary" data-role="h4">h=4</button>' +
        '<button type="button" class="wg-button" data-role="h8">h=8</button>' +
      '</div>' +
      '<div class="mb-row">' +
        '<div class="mb-col">' +
          '<div class="wg-label"><span>决策边界 p(圆内)</span><span>[−1.6, 1.6]²</span></div>' +
          '<canvas class="wg-canvas" data-role="map"></canvas>' +
          '<p class="wg-note">虚线圆=真实单位圆；斜虚线=各隐藏神经元的 ReLU 折线；红圈=分错的点。</p>' +
        '</div>' +
        '<div class="mb-col">' +
          '<div class="wg-label"><span>网络 2-h-2</span><span>点击边/偏置</span></div>' +
          '<div class="mb-svgwrap" data-role="net"></div>' +
          '<p class="wg-note">蓝边=正权重，红边=负权重，粗细=|w|。点击边 +0.3，Shift+点击 −0.3；隐藏/输出节点下方的偏置同样可点。</p>' +
        '</div>' +
      '</div>' +
      '<div class="wg-readout">' +
        '<div class="wg-stat"><span>训练点准确率</span><b data-role="acc"></b></div>' +
        '<div class="wg-stat"><span>分对的点</span><b data-role="okn"></b></div>' +
        '<div class="wg-stat"><span>活跃 ReLU 折线</span><b data-role="folds"></b></div>' +
        '<div class="wg-stat"><span>结构</span><b data-role="arch">2-4-2</b></div>' +
      '</div>' +
      '<p class="wg-note" data-role="foldnote"></p>' +
      '<div class="wg-controls">' +
        '<button type="button" class="wg-button is-primary" data-role="btn-random">随机权重</button>' +
        '<button type="button" class="wg-button" data-role="btn-line">预设 · 近似直线</button>' +
        '<button type="button" class="wg-button" data-role="btn-circle">预设 · 围成圆</button>' +
        '<button type="button" class="wg-button" data-role="btn-reset">重置</button>' +
      '</div>';

    var style = document.createElement('style');
    style.textContent =
      '.mb-row { display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-start; }' +
      '.mb-col { flex: 1 1 272px; min-width: 250px; }' +
      '.mb-svgwrap svg { display: block; width: 100%; height: auto; user-select: none; }' +
      '.mb-edge { cursor: pointer; }' +
      '.mb-edge .mb-hit { stroke: transparent; stroke-width: 16; fill: none; }' +
      '.mb-edge .mb-vis { fill: none; }' +
      '.mb-edge:hover .mb-vis { stroke-width: 3.5; }' +
      '.mb-edge:hover .mb-wrect { fill: #c8ff47; }' +
      '.mb-chip { cursor: pointer; }' +
      '.mb-chip:hover .mb-chiprect { fill: #c8ff47; stroke: #071321; }';
    container.appendChild(style);

    var DATA = genData();
    var state = { model: presetLine(4), preset: 'line' };
    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };
    var mapCanvas = q('map');
    var netBox = q('net');

    /* ---------- 边界热图 ---------- */
    function drawMap() {
      var wrap = mapCanvas.parentElement;
      var cssW = Math.max(200, Math.min(wrap ? wrap.clientWidth : 300, 330));
      var ratio = window.devicePixelRatio || 1;
      mapCanvas.style.width = cssW + 'px';
      mapCanvas.style.height = cssW + 'px';
      mapCanvas.style.maxWidth = '100%';
      mapCanvas.style.margin = '0 auto';
      mapCanvas.width = Math.round(cssW * ratio);
      mapCanvas.height = Math.round(cssW * ratio);
      var ctx = mapCanvas.getContext('2d');
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      var W = cssW, H = cssW;

      // 60x60 概率网格 -> 离屏画布 -> 平滑放大
      var N = 60;
      var off = document.createElement('canvas');
      off.width = N; off.height = N;
      var octx = off.getContext('2d');
      var img = octx.createImageData(N, N);
      for (var i = 0; i < N; i++) {
        for (var j = 0; j < N; j++) {
          var x = -DOM + 2 * DOM * i / (N - 1);
          var y = DOM - 2 * DOM * j / (N - 1);
          var t = 1 / (1 + Math.exp(-forward(state.model, x, y).d * 0.9));
          var k = (j * N + i) * 4;
          img.data[k] = Math.round(242 + (11 - 242) * t);
          img.data[k + 1] = Math.round(233 + (99 - 233) * t);
          img.data[k + 2] = Math.round(214 + (243 - 214) * t);
          img.data[k + 3] = 255;
        }
      }
      octx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.clearRect(0, 0, W, H);
      ctx.drawImage(off, 0, 0, W, H);

      var sx = function (x) { return (x + DOM) / (2 * DOM) * W; };
      var sy = function (y) { return H - (y + DOM) / (2 * DOM) * H; };

      // 网格参考线
      ctx.strokeStyle = 'rgba(7,19,33,.14)'; ctx.lineWidth = 1;
      [-1, 0, 1].forEach(function (g) {
        ctx.beginPath(); ctx.moveTo(sx(g), 0); ctx.lineTo(sx(g), H); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, sy(g)); ctx.lineTo(W, sy(g)); ctx.stroke();
      });
      ctx.fillStyle = '#65717c'; ctx.font = '9px ui-monospace, monospace';
      ctx.fillText('+1', sx(1) - 10, sy(1) - 4);
      ctx.fillText('−1', sx(-1) + 3, sy(1) - 4);
      ctx.fillText('x₁', W - 16, sy(0) - 5);
      ctx.fillText('x₂', sx(0) + 5, 12);

      // 真实单位圆
      ctx.strokeStyle = 'rgba(7,19,33,.75)'; ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.arc(sx(0), sy(0), (1 / (2 * DOM)) * W, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);

      // ReLU 折线：w·x + b = 0 在框内的线段（slab 裁剪）
      ctx.strokeStyle = 'rgba(7,19,33,.4)'; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
      for (var u = 0; u < state.model.h; u++) {
        var w0 = state.model.w1[u][0], w1 = state.model.w1[u][1], b = state.model.b1[u];
        var n2 = w0 * w0 + w1 * w1;
        if (n2 < 0.02) { continue; }               // 近零权重的折线远在框外
        var px = -b * w0 / n2, py = -b * w1 / n2;  // 直线上离原点最近的点
        var dx = -w1 / Math.sqrt(n2), dy = w0 / Math.sqrt(n2);
        var tLo = -1e9, tHi = 1e9;
        [[dx, px, -DOM, DOM], [dy, py, -DOM, DOM]].forEach(function (s) {
          var d = s[0], p = s[1], lo = s[2], hi = s[3];
          if (Math.abs(d) < 1e-9) { return; }
          var t1 = (lo - p) / d, t2 = (hi - p) / d;
          tLo = Math.max(tLo, Math.min(t1, t2));
          tHi = Math.min(tHi, Math.max(t1, t2));
        });
        if (tHi > tLo) {
          ctx.beginPath();
          ctx.moveTo(sx(px + tLo * dx), sy(py + tLo * dy));
          ctx.lineTo(sx(px + tHi * dx), sy(py + tHi * dy));
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);

      // 数据点（分错的套红圈）
      DATA.forEach(function (p) {
        var pred = forward(state.model, p.x, p.y).d > 0 ? 1 : 0;
        ctx.beginPath();
        ctx.arc(sx(p.x), sy(p.y), 3.4, 0, Math.PI * 2);
        if (p.c === 1) {
          ctx.fillStyle = BLUE; ctx.fill();
          ctx.strokeStyle = PAPER; ctx.lineWidth = 1; ctx.stroke();
        } else {
          ctx.fillStyle = PAPER; ctx.fill();
          ctx.strokeStyle = INK; ctx.lineWidth = 1; ctx.stroke();
        }
        if (pred !== p.c) {
          ctx.beginPath();
          ctx.arc(sx(p.x), sy(p.y), 6, 0, Math.PI * 2);
          ctx.strokeStyle = RED; ctx.lineWidth = 1.6; ctx.stroke();
        }
      });
    }

    /* ---------- 网络图（每次整体重建） ---------- */
    function edgeGroup(svgParent, x1, y1, x2, y2, w, onClick) {
      var g = el('g', { 'class': 'mb-edge' }, svgParent);
      var vis = Math.abs(w);
      el('line', { x1: x1, y1: y1, x2: x2, y2: y2, 'class': 'mb-hit' }, g);
      el('line', {
        x1: x1, y1: y1, x2: x2, y2: y2, 'class': 'mb-vis',
        stroke: w >= 0 ? BLUE : RED,
        'stroke-width': Math.max(0.8, Math.min(4, 0.7 + vis * 1.1)),
      }, g);
      var mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
      var label = fmt(w);
      var wRect = Math.max(20, label.length * 5.6 + 6);
      el('rect', {
        x: mx - wRect / 2, y: my - 6, width: wRect, height: 11,
        'class': 'mb-wrect', fill: PAPER, stroke: 'rgba(7,19,33,.35)', 'stroke-width': 0.6,
      }, g);
      var txt = el('text', {
        x: mx, y: my + 3, 'text-anchor': 'middle',
        'font-size': 7.5, 'font-family': 'ui-monospace, monospace',
        fill: w >= 0 ? BLUE : RED, 'font-weight': 700,
      }, g);
      txt.textContent = label;
      var title = el('title', {}, g);
      title.textContent = '权重 ' + label + '：点击 +0.3 / Shift+点击 −0.3';
      g.addEventListener('click', function (ev) { onClick(ev.shiftKey ? -STEP : STEP); });
      return g;
    }

    function chip(svgParent, cx, cy, prefix, v, onClick) {
      var g = el('g', { 'class': 'mb-chip' }, svgParent);
      var label = prefix + ' ' + fmtB(v);
      var wRect = label.length * 5.4 + 8;
      el('rect', {
        x: cx - wRect / 2, y: cy - 8, width: wRect, height: 11,
        'class': 'mb-chiprect', fill: '#f4f2ec', stroke: 'rgba(7,19,33,.4)', 'stroke-width': 0.8,
      }, g);
      var txt = el('text', {
        x: cx, y: cy + 1, 'text-anchor': 'middle',
        'font-size': 7.5, 'font-family': 'ui-monospace, monospace', fill: INK, 'font-weight': 700,
      }, g);
      txt.textContent = label;
      var title = el('title', {}, g);
      title.textContent = '偏置 ' + fmtB(v) + '：点击 +0.3 / Shift+点击 −0.3';
      g.addEventListener('click', function (ev) { onClick(ev.shiftKey ? -STEP : STEP); });
    }

    function node(svgParent, cx, cy, r, fill, label, light) {
      el('circle', { cx: cx, cy: cy, r: r, fill: fill, stroke: INK, 'stroke-width': 1.4 }, svgParent);
      if (label) {
        var t = el('text', {
          x: cx, y: cy + 2.6, 'text-anchor': 'middle', 'font-size': 8.5,
          'font-family': 'ui-monospace, monospace', fill: light ? PAPER : INK, 'font-weight': 700,
        }, svgParent);
        t.textContent = label;
      }
    }

    function buildNet() {
      netBox.innerHTML = '';
      var m = state.model;
      var VBW = 300, VBH = 330;
      var svg = el('svg', { viewBox: '0 0 ' + VBW + ' ' + VBH, role: 'img' }, netBox);

      var capY = 16;
      [[40, '输入 x'], [150, '隐藏 ReLU'], [258, '输出']].forEach(function (c) {
        var t = el('text', {
          x: c[0], y: capY, 'text-anchor': 'middle', 'font-size': 8,
          'font-family': 'ui-monospace, monospace', fill: '#65717c', 'font-weight': 700,
          'letter-spacing': '0.5',
        }, svg);
        t.textContent = c[1];
      });

      var gap = m.h <= 2 ? 76 : (m.h <= 4 ? 56 : 34);
      var rH = m.h >= 8 ? 10 : (m.h >= 4 ? 12 : 14);
      var hCy = [];
      for (var j = 0; j < m.h; j++) { hCy.push(165 - gap * (m.h - 1) / 2 + gap * j); }
      var inCy = [132, 198], outCy = [132, 198], rIO = 13;

      // 边：输入 -> 隐藏
      for (var a = 0; a < 2; a++) {
        for (var b = 0; b < m.h; b++) {
          (function (ai, bj) {
            var stagger = ai === 0 ? 0.42 : 0.58;
            var x1 = 40, y1 = inCy[ai], x2 = 150, y2 = hCy[bj];
            var mx = x1 + (x2 - x1) * stagger, my = y1 + (y2 - y1) * (0.5 + (ai === 0 ? -0.06 : 0.06));
            // 自定义中点：手动放标签，line 仍连节点
            var g = edgeGroup(svg, x1, y1, x2, y2, m.w1[bj][ai], function (d) {
              m.w1[bj][ai] = clampW(m.w1[bj][ai] + d); render();
            });
            // 把标签挪到交错位置，避免 h=8 时重叠
            var rect = g.querySelector('.mb-wrect'), txt = g.querySelector('text');
            var half = parseFloat(rect.getAttribute('width')) / 2;
            rect.setAttribute('x', mx - half);
            rect.setAttribute('y', my - 5.5);
            txt.setAttribute('x', mx);
            txt.setAttribute('y', my + 3);
          })(a, b);
        }
      }

      // 边：隐藏 -> 输出
      for (var c = 0; c < m.h; c++) {
        for (var d2 = 0; d2 < 2; d2++) {
          (function (cj, ok) {
            var stagger = ok === 0 ? 0.42 : 0.58;
            var x1 = 150, y1 = hCy[cj], x2 = 258, y2 = outCy[ok];
            var mx = x1 + (x2 - x1) * stagger, my = y1 + (y2 - y1) * 0.5;
            var g = edgeGroup(svg, x1, y1, x2, y2, m.w2[ok][cj], function (d) {
              m.w2[ok][cj] = clampW(m.w2[ok][cj] + d); render();
            });
            var rect = g.querySelector('.mb-wrect'), txt = g.querySelector('text');
            var half = parseFloat(rect.getAttribute('width')) / 2;
            rect.setAttribute('x', mx - half);
            txt.setAttribute('x', mx);
            txt.setAttribute('y', my + 3);
          })(c, d2);
        }
      }

      // 节点
      node(svg, 40, inCy[0], rIO, PAPER, 'x₁');
      node(svg, 40, inCy[1], rIO, PAPER, 'x₂');
      for (var n = 0; n < m.h; n++) { node(svg, 150, hCy[n], rH, PAPER, String(n + 1)); }
      node(svg, 258, outCy[0], rIO, BEIGE, '0');
      node(svg, 258, outCy[1], rIO, BLUE, '1', true);

      // 输出类别说明
      var lab0 = el('text', { x: 258, y: outCy[0] + rIO + 13, 'text-anchor': 'middle', 'font-size': 7.5, 'font-family': 'ui-monospace, monospace', fill: INK, 'font-weight': 700 }, svg);
      lab0.textContent = '圆外 0';
      var lab1 = el('text', { x: 258, y: outCy[1] + rIO + 13, 'text-anchor': 'middle', 'font-size': 7.5, 'font-family': 'ui-monospace, monospace', fill: BLUE, 'font-weight': 700 }, svg);
      lab1.textContent = '圆内 1';

      // 偏置芯片：隐藏节点下方、输出节点下方再靠下
      for (var p = 0; p < m.h; p++) {
        (function (pj) {
          chip(svg, 150, hCy[pj] + rH + 12, 'b', m.b1[pj], function (dd) {
            m.b1[pj] = clampW(m.b1[pj] + dd); render();
          });
        })(p);
      }
      chip(svg, 258, outCy[0] + rIO + 26, 'b', m.b2[0], function (dd) {
        m.b2[0] = clampW(m.b2[0] + dd); render();
      });
      chip(svg, 258, outCy[1] + rIO + 26, 'b', m.b2[1], function (dd) {
        m.b2[1] = clampW(m.b2[1] + dd); render();
      });
    }

    /* ---------- 读数 ---------- */
    function activeFolds() {
      var n = 0;
      for (var j = 0; j < state.model.h; j++) {
        var w = state.model.w1[j];
        if (Math.hypot(w[0], w[1]) > 0.15) { n++; }
      }
      return n;
    }

    function updateStats() {
      var ok = 0;
      DATA.forEach(function (p) {
        if ((forward(state.model, p.x, p.y).d > 0 ? 1 : 0) === p.c) { ok++; }
      });
      q('acc').textContent = (ok / DATA.length * 100).toFixed(1) + '%';
      q('okn').textContent = ok + '/' + DATA.length;
      var act = activeFolds();
      q('folds').textContent = act + '/' + state.model.h;
      var arch = '2-' + state.model.h + '-2';
      q('arch').textContent = arch;
      q('hval').textContent = arch;

      var tail;
      if (state.model.h <= 2) {
        tail = '只有 2 条折线，无论怎么调也围不出闭合区域——这就是宽度不够时 MLP 表达能力的天花板。';
      } else if (state.model.h <= 4) {
        tail = '4 条折线首尾相接，围出的是四边形：四个角盖不住圆外远的点，四个边中段又切进圆内。';
      } else {
        tail = '8 条折线首尾相接，围出近似圆的八边形——折线越多，边界越接近圆。';
      }
      q('foldnote').textContent = 'ReLU 折线：第 j 个隐藏神经元在输入平面留下直线 w·x+b=0（左图斜虚线），折线的一侧输出恒为 0。当前活跃 ' + act + '/' + state.model.h + ' 条。' + tail;
    }

    function render() {
      drawMap();
      buildNet();
      updateStats();
    }

    /* ---------- 控件 ---------- */
    function setH(h) {
      state.model = state.preset === 'circle' ? presetCircle(h)
        : state.preset === 'random' ? randomModel(h)
        : presetLine(h);
      ['2', '4', '8'].forEach(function (n) {
        var btn = q('h' + n);
        btn.classList.toggle('is-primary', String(h) === n);
      });
      render();
    }

    q('h2').addEventListener('click', function () { setH(2); });
    q('h4').addEventListener('click', function () { setH(4); });
    q('h8').addEventListener('click', function () { setH(8); });
    q('btn-random').addEventListener('click', function () {
      state.preset = 'random';
      state.model = randomModel(state.model.h);
      render();
    });
    q('btn-line').addEventListener('click', function () {
      state.preset = 'line';
      state.model = presetLine(state.model.h);
      render();
    });
    q('btn-circle').addEventListener('click', function () {
      state.preset = 'circle';
      state.model = presetCircle(state.model.h);
      render();
    });
    q('btn-reset').addEventListener('click', function () {
      state.preset = 'line';
      setH(4);
    });

    var onResize = function () { render(); };
    window.addEventListener('resize', onResize);
    render();

    return function dispose() {
      window.removeEventListener('resize', onResize);
    };
  }

  window.DLWidget.register('mlp-boundary', mount);
})();
