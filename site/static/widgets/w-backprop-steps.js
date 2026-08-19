/* backprop-steps — 固定 2-4-4-2 网络、固定样本 (0.8,0.3)，15 步拆开一次前向 + 损失 + 反向 + 更新。
 *
 * 所有数值在加载时按同一套公式算好存进 STEPS（渲染时不再计算）：
 * 前向 6 步（z1→h1→z2→h2→logits→P）+ 损失 1 步 + 反向 7 步 + 更新 1 步，η=0.3。
 * 权重刻意选成一位小数，方便读者用面板里的算式手算复核。
 */
(function () {
  'use strict';

  /* ---------- 固定设定 ---------- */
  var X = [0.8, 0.3], Y = 0, ETA = 0.3;
  var W1 = [[0.8, -0.5], [-0.6, 0.9], [0.5, 0.5], [1.0, -1.0]], B1 = [0.1, -0.4, -0.2, 0];
  var W2 = [[0.6, 0, -0.4, 0.3], [-0.5, 0.7, 0.2, -0.6], [0.4, -0.3, 0.8, 0.2], [0.3, 0.5, -0.2, 0.7]], B2 = [-0.3, 0.2, -0.1, 0.05];
  var W3 = [[0.5, -0.9, 0.4, -0.5], [-0.6, 1.0, -0.5, 0.7]], B3 = [-0.1, 0.1];

  var relu = function (v) { return v > 0 ? v : 0; };
  var mm = function (W, v) { return W.map(function (r) { return r.reduce(function (s, w, j) { return s + w * v[j]; }, 0); }); };
  var addv = function (v, b) { return v.map(function (x, i) { return x + b[i]; }); };

  function forward(w1, b1, w2, b2, w3, b3) {
    var z1 = addv(mm(w1, X), b1), h1 = z1.map(relu);
    var z2 = addv(mm(w2, h1), b2), h2 = z2.map(relu);
    var lg = addv(mm(w3, h2), b3);
    var mx = Math.max.apply(null, lg);
    var ex = lg.map(function (v) { return Math.exp(v - mx); });
    var s = ex.reduce(function (a, b) { return a + b; }, 0);
    var P = ex.map(function (v) { return v / s; });
    return { z1: z1, h1: h1, z2: z2, h2: h2, lg: lg, P: P, L: -Math.log(P[Y]) };
  }

  var F = forward(W1, B1, W2, B2, W3, B3);           // 初始前向
  var dlogits = F.P.map(function (p, i) { return p - (i === Y ? 1 : 0); });
  var dW3 = dlogits.map(function (d) { return F.h2.map(function (h) { return d * h; }); });
  var db3 = dlogits.slice();
  var dh2 = F.h2.map(function (_, j) { return dlogits.reduce(function (s, d, i) { return s + W3[i][j] * d; }, 0); });
  var dz2 = dh2.map(function (d, j) { return F.z2[j] > 0 ? d : 0; });
  var dW2 = dz2.map(function (d) { return F.h1.map(function (h) { return d * h; }); });
  var db2 = dz2.slice();
  var dh1 = F.h1.map(function (_, j) { return dz2.reduce(function (s, d, i) { return s + W2[i][j] * d; }, 0); });
  var dz1 = dh1.map(function (d, j) { return F.z1[j] > 0 ? d : 0; });
  var dW1 = dz1.map(function (d) { return X.map(function (x) { return d * x; }); });
  var db1 = dz1.slice();
  var upd = function (W, dW) { return W.map(function (r, i) { return r.map(function (w, j) { return w - ETA * dW[i][j]; }); }); };
  var G = forward(upd(W1, dW1), B1.map(function (v, i) { return v - ETA * db1[i]; }),
                  upd(W2, dW2), B2.map(function (v, i) { return v - ETA * db2[i]; }),
                  upd(W3, dW3), B3.map(function (v, i) { return v - ETA * db3[i]; }));   // 更新后前向
  var NPARAM = 8 + 4 + 16 + 4 + 8 + 2;

  /* ---------- 显示用的记号与格式化 ---------- */
  var SB = ['₁', '₂', '₃', '₄'], SO = ['₀', '₁'];
  var z1n = function (i) { return 'z¹' + SB[i]; }, h1n = function (i) { return 'h¹' + SB[i]; };
  var z2n = function (i) { return 'z²' + SB[i]; }, h2n = function (i) { return 'h²' + SB[i]; };

  function fmt(v, d) {
    if (d == null) { d = 2; }
    if (Math.abs(v) < 0.0005) { return '0'; }
    var s = Math.abs(v).toFixed(d).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
    return (v < 0 ? '−' : '') + s;
  }
  function fmtP(v) { return v.toFixed(3); }   // 概率保留 3 位
  function wt(w) { return w < 0 ? '(' + fmt(w, 1) + ')' : fmt(w, 1); }        // 权重（负数带括号）
  function dot(w, v) { return fmt(w, 1) + '·' + fmt(v); }                     // 一项乘积

  /* 面板元素 */
  function ar(name, terms, b, res, cls) {
    var bs = (b == null ? '' : (b < 0 ? ' − ' + fmt(-b, 1) : ' + ' + fmt(b, 1)));
    var body = '';
    terms.forEach(function (t, i) {
      if (i === 0) { body += t; }
      else if (t.charAt(0) === '−') { body += ' − ' + t.slice(1); }
      else { body += ' + ' + t; }
    });
    return '<div class="bp-line"><i>' + name + '</i> = ' + body + bs +
      ' <b class="' + (cls || '') + '">= ' + res + '</b></div>';
  }
  function kv(k, v, cls) {
    return '<div class="bp-kv"><span>' + k + '</span><b class="' + (cls || '') + '">' + v + '</b></div>';
  }
  function mat(title, rowLab, rows, note) {
    var n = rows[0].length, max = 0, i, j;
    for (i = 0; i < rows.length; i++) { for (j = 0; j < n; j++) { if (Math.abs(rows[i][j]) > max) { max = Math.abs(rows[i][j]); } } }
    var html = '<div class="bp-mh">' + title + '</div><div class="bp-mat" style="grid-template-columns:' + (rowLab ? '30px ' : '') + 'repeat(' + n + ',1fr)">';
    if (rowLab) {
      html += '<span class="bp-ml"></span>';
      for (j = 0; j < n; j++) { html += '<span class="bp-ml">' + (j + 1) + '</span>'; }
    }
    for (i = 0; i < rows.length; i++) {
      if (rowLab) { html += '<span class="bp-ml">' + rowLab[i] + '</span>'; }
      for (j = 0; j < n; j++) {
        var g = rows[i][j], dead = Math.abs(g) < 0.0005, bg = '';
        if (!dead) {
          var a = 0.07 + 0.32 * Math.abs(g) / (max || 1);
          bg = ' style="background:' + (g < 0 ? 'rgba(192,57,43,' + a.toFixed(2) + ')' : 'rgba(11,99,243,' + a.toFixed(2) + ')') + '"';
        }
        html += '<span class="bp-mc' + (dead ? ' z' : '') + '"' + bg + '>' + fmt(g) + '</span>';
      }
    }
    html += '</div>';
    if (note) { html += '<p class="bp-mnote">' + note + '</p>'; }
    return html;
  }

  /* ---------- 15 步定义 ---------- */
  var STEPS = [
    { phase: 'f', chip: '前向 1/6', title: 'z¹ = W¹·x + b¹',
      desc: '第一层线性变换：4 个神经元各算一个加权和。',
      formula: 'z¹ = W¹·x + b¹', shape: '(4×2)·(2×1) + (4×1) → (4×1)',
      legend: '节点数值 = z¹　·　绿 = 本步新算出',
      panel:
        ar(z1n(0), [dot(W1[0][0], X[0]), dot(W1[0][1], X[1])], B1[0], fmt(F.z1[0]), 'g') +
        ar(z1n(1), [dot(W1[1][0], X[0]), dot(W1[1][1], X[1])], B1[1], fmt(F.z1[1])) +
        ar(z1n(2), [dot(W1[2][0], X[0]), dot(W1[2][1], X[1])], B1[2], fmt(F.z1[2])) +
        ar(z1n(3), [dot(W1[3][0], X[0]), dot(W1[3][1], X[1])], B1[3], fmt(F.z1[3])),
      g: { nodes: { 0: { vals: X, st: 'known' }, 1: { vals: F.z1, st: 'new' } }, edges: { 0: { mode: 'fwd' } } } },

    { phase: 'f', chip: '前向 2/6', title: 'h¹ = ReLU(z¹)',
      desc: '负的 z 归零：z¹₂ < 0，第 2 个神经元输出 0。',
      formula: 'h¹ = max(0, z¹)', shape: '(4×1) → (4×1)',
      legend: '灰节点 = ReLU 截断（z ≤ 0 → 0）',
      panel:
        kv('max(0, ' + fmt(F.z1[0]) + ')', fmt(F.h1[0]), 'g') +
        kv('max(0, ' + fmt(F.z1[1]) + ')', '0　被截断', 'r') +
        kv('max(0, ' + fmt(F.z1[2]) + ')', fmt(F.h1[2]), 'g') +
        kv('max(0, ' + fmt(F.z1[3]) + ')', fmt(F.h1[3]), 'g'),
      g: { nodes: { 0: { vals: X, st: 'known' }, 1: { vals: F.h1, st: 'new', dead: F.z1.map(function (v) { return v <= 0; }) } } } },

    { phase: 'f', chip: '前向 3/6', title: 'z² = W²·h¹ + b²',
      desc: '第二层加权和；h¹₂ = 0，第 2 列权重不参与。',
      formula: 'z² = W²·h¹ + b²', shape: '(4×4)·(4×1) + (4×1) → (4×1)',
      legend: '节点数值 = z²　·　左边灰色 0 = 上一步截断的 h¹₂',
      panel:
        ar(z2n(0), W2[0].map(function (w, j) { return dot(w, F.h1[j]); }), B2[0], fmt(F.z2[0]), 'g') +
        ar(z2n(1), W2[1].map(function (w, j) { return dot(w, F.h1[j]); }), B2[1], fmt(F.z2[1])) +
        ar(z2n(2), W2[2].map(function (w, j) { return dot(w, F.h1[j]); }), B2[2], fmt(F.z2[2])) +
        ar(z2n(3), W2[3].map(function (w, j) { return dot(w, F.h1[j]); }), B2[3], fmt(F.z2[3])),
      g: { nodes: { 0: { vals: X, st: 'known' }, 1: { vals: F.h1, st: 'known', dead: F.z1.map(function (v) { return v <= 0; }) }, 2: { vals: F.z2, st: 'new' } }, edges: { 1: { mode: 'fwd' } } } },

    { phase: 'f', chip: '前向 4/6', title: 'h² = ReLU(z²)',
      desc: 'z²₂ < 0 同样被截断，第 2 个单元两层输出均为 0。',
      formula: 'h² = max(0, z²)', shape: '(4×1) → (4×1)',
      legend: '灰节点 = ReLU 截断',
      panel:
        kv('max(0, ' + fmt(F.z2[0]) + ')', fmt(F.h2[0]), 'g') +
        kv('max(0, ' + fmt(F.z2[1]) + ')', '0　被截断', 'r') +
        kv('max(0, ' + fmt(F.z2[2]) + ')', fmt(F.h2[2]), 'g') +
        kv('max(0, ' + fmt(F.z2[3]) + ')', fmt(F.h2[3]), 'g'),
      g: { nodes: { 0: { vals: X, st: 'known' }, 1: { vals: F.h1, st: 'known', dead: F.z1.map(function (v) { return v <= 0; }) }, 2: { vals: F.h2, st: 'new', dead: F.z2.map(function (v) { return v <= 0; }) } } } },

    { phase: 'f', chip: '前向 5/6', title: 'logits = W³·h² + b³',
      desc: '输出层：h²₂ = 0，该列权重不参与。',
      formula: 'logits = W³·h² + b³', shape: '(2×4)·(4×1) + (2×1) → (2×1)',
      legend: '节点数值 = logits　·　下标签 = 类别',
      panel:
        ar('logit' + SO[0], W3[0].map(function (w, j) { return dot(w, F.h2[j]); }), B3[0], fmt(F.lg[0]), 'g') +
        ar('logit' + SO[1], W3[1].map(function (w, j) { return dot(w, F.h2[j]); }), B3[1], fmt(F.lg[1]), 'g') +
        kv('比较', 'logit' + SO[1] + ' 大 → 猜「圆外」', 'r'),
      g: { nodes: { 0: { vals: X, st: 'known' }, 1: { vals: F.h1, st: 'known', dead: F.z1.map(function (v) { return v <= 0; }) }, 2: { vals: F.h2, st: 'known', dead: F.z2.map(function (v) { return v <= 0; }) }, 3: { vals: F.lg, st: 'new' } }, edges: { 2: { mode: 'fwd' } } } },

    { phase: 'f', chip: '前向 6/6', title: 'P = softmax(logits)',
      desc: 'P₀ = 0.432 < 0.5，该点被猜成圆外。',
      formula: 'P = softmax(logits)', shape: 'e^z ÷ Σ e^z → (2×1)',
      legend: '节点半径 ∝ 概率　·　数值 = P',
      panel:
        kv('e^(logit' + SO[0] + ' − ' + fmt(Math.max(F.lg[0], F.lg[1])) + ')', fmt(Math.exp(F.lg[0] - Math.max(F.lg[0], F.lg[1]))) + '　（减 max 防溢出）') +
        kv('e^(logit' + SO[1] + ' − ' + fmt(Math.max(F.lg[0], F.lg[1])) + ')', '1　（max 自己 = 1）') +
        kv('P' + SO[0] + ' = 0.76 ÷ 1.76', fmtP(F.P[0]), 'g') +
        kv('P' + SO[1] + ' = 1 ÷ 1.76', fmtP(F.P[1])) +
        kv('校验 ΣP', fmtP(F.P[0] + F.P[1])),
      g: { nodes: { 0: { vals: X, st: 'known' }, 1: { vals: F.h1, st: 'known', dead: F.z1.map(function (v) { return v <= 0; }) }, 2: { vals: F.h2, st: 'known', dead: F.z2.map(function (v) { return v <= 0; }) }, 3: { vals: F.P, st: 'new' } }, radius: { 3: [12 + 9 * F.P[0], 12 + 9 * F.P[1]] } } },

    { phase: 'l', chip: '损失 1/1', title: 'L = −log P[y]',
      desc: 'P₀ = 0.432，损失 L = 0.839。',
      formula: 'L = −log P₀', shape: '标量',
      legend: '节点数值 = P　·　求损失不改任何节点',
      panel:
        ar('L', ['−log(' + fmtP(F.P[0]) + ')'], null, fmt(F.L, 3), 'g') +
        kv('P₀（正确类）', fmtP(F.P[0]), 'r') +
        kv('P₁（错误类）', fmtP(F.P[1])) +
        kv('直觉', 'P₀=1 → L=0；P₀=0.5 → L≈0.69'),
      g: { nodes: { 0: { vals: X, st: 'known' }, 1: { vals: F.h1, st: 'known', dead: F.z1.map(function (v) { return v <= 0; }) }, 2: { vals: F.h2, st: 'known', dead: F.z2.map(function (v) { return v <= 0; }) }, 3: { vals: F.P, st: 'known' } }, radius: { 3: [12 + 9 * F.P[0], 12 + 9 * F.P[1]] } } },

    { phase: 'b', chip: '反向 1/7', title: 'dlogits = (P − Y)/m',
      desc: 'dlogits = P − Y：正确类为负，错误类为正。',
      formula: 'dlogits = (P − Y) / m', shape: '(2×1)，m = 1',
      legend: '节点数值 = dlogits（红 = 梯度）',
      panel:
        ar('dlogits' + SO[0], ['(' + fmtP(F.P[0]) + ' − 1) / 1'], null, fmt(dlogits[0]), 'g') +
        ar('dlogits' + SO[1], ['(' + fmtP(F.P[1]) + ' − 0) / 1'], null, fmt(dlogits[1]), 'g') +
        kv('校验 Σdlogits', '0　（softmax 的性质）'),
      g: { nodes: { 0: { vals: X, st: 'known' }, 1: { vals: F.h1, st: 'known', dead: F.z1.map(function (v) { return v <= 0; }) }, 2: { vals: F.h2, st: 'known', dead: F.z2.map(function (v) { return v <= 0; }) }, 3: { vals: dlogits, st: 'grad' } } } },

    { phase: 'b', chip: '反向 2/7', title: 'dW³ = dlogits·(h²)ᵀ，db³ = Σdlogits',
      desc: 'dW³[i][j] = dlogits[i]·h²[j]；h²₂ = 0，第 2 列全 0。',
      formula: 'dW³ = dlogits · (h²)ᵀ，db³ = Σ dlogits', shape: '(2×1)·(1×4) → (2×4)',
      legend: '边上的红字 = 该连接的梯度 dW³　·　越粗 |梯度| 越大',
      panel:
        mat('dW³（行 = 输出，列 = h² 单元）', ['ŷ₀', 'ŷ₁'], dW3, '第 2 列全 0：h²₂ = 0，权重收不到梯度。') +
        kv('db³', '[' + fmt(db3[0]) + ', ' + fmt(db3[1]) + ']'),
      g: { nodes: { 0: { vals: X, st: 'known' }, 1: { vals: F.h1, st: 'known', dead: F.z1.map(function (v) { return v <= 0; }) }, 2: { vals: F.h2, st: 'known', dead: F.z2.map(function (v) { return v <= 0; }) }, 3: { vals: dlogits, st: 'grad' } }, edges: { 2: { mode: 'bwd', mat: dW3, labels: true } } } },

    { phase: 'b', chip: '反向 3/7', title: 'dh² = (W³)ᵀ·dlogits',
      desc: '每个 h² 单元收到的梯度 = W³ 对应列的加权和。',
      formula: 'dh² = (W³)ᵀ · dlogits', shape: '(4×2)·(2×1) → (4×1)',
      legend: '节点数值 = dh²（红 = 梯度）　·　虚线边 = 梯度流过的连接',
      panel:
        ar('dh²' + SB[0], [dot(W3[0][0], dlogits[0]), dot(W3[1][0], dlogits[1])], null, fmt(dh2[0]), 'g') +
        ar('dh²' + SB[1], [dot(W3[0][1], dlogits[0]), dot(W3[1][1], dlogits[1])], null, fmt(dh2[1]), 'g') +
        ar('dh²' + SB[2], [dot(W3[0][2], dlogits[0]), dot(W3[1][2], dlogits[1])], null, fmt(dh2[2]), 'g') +
        ar('dh²' + SB[3], [dot(W3[0][3], dlogits[0]), dot(W3[1][3], dlogits[1])], null, fmt(dh2[3]), 'g'),
      g: { nodes: { 0: { vals: X, st: 'known' }, 1: { vals: F.h1, st: 'known', dead: F.z1.map(function (v) { return v <= 0; }) }, 2: { vals: dh2, st: 'grad' }, 3: { vals: dlogits, st: 'grad' } }, edges: { 2: { mode: 'bwd' } } } },

    { phase: 'b', chip: '反向 4/7', title: 'dz² = dh² ⊙ 1[z² > 0]',
      desc: 'z² ≤ 0 的单元梯度归零；dh²₂ = 1.08 也被清零。',
      formula: 'dz² = dh² ⊙ 1[z² > 0]', shape: '(4×1) 逐元素',
      legend: '灰节点 = 梯度被 ReLU 截断（→ 0）',
      panel:
        kv('1[' + fmt(F.z2[0]) + ' > 0] × ' + fmt(dh2[0]), fmt(dz2[0]), 'g') +
        kv('1[' + fmt(F.z2[1]) + ' > 0] × ' + fmt(dh2[1]), '0　被截断', 'r') +
        kv('1[' + fmt(F.z2[2]) + ' > 0] × ' + fmt(dh2[2]), fmt(dz2[2]), 'g') +
        kv('1[' + fmt(F.z2[3]) + ' > 0] × ' + fmt(dh2[3]), fmt(dz2[3]), 'g'),
      g: { nodes: { 0: { vals: X, st: 'known' }, 1: { vals: F.h1, st: 'known', dead: F.z1.map(function (v) { return v <= 0; }) }, 2: { vals: dz2, st: 'grad', dead: F.z2.map(function (v) { return v <= 0; }) }, 3: { vals: dlogits, st: 'grad' } } } },

    { phase: 'b', chip: '反向 5/7', title: 'dW² = dz²·(h¹)ᵀ，db² = Σdz²',
      desc: 'dW²[i][j] = dz²[i]·h¹[j]；第 2 行与第 2 列全 0。',
      formula: 'dW² = dz² · (h¹)ᵀ，db² = Σ dz²', shape: '(4×1)·(1×4) → (4×4)',
      legend: '边上的红字 = dW²（只标 |梯度| 最大的 8 条，完整数值见右）　·　灰虚线 = |梯度|≈0',
      panel:
        mat('dW²（行 = z² 单元，列 = h¹ 单元）', ['1', '2', '3', '4'], dW2, '一行全 0 = 该单元梯度被截断；一列全 0 = 该输入为 0。') +
        kv('db²', '[' + dz2.map(function (v) { return fmt(v); }).join(', ') + ']'),
      g: { nodes: { 0: { vals: X, st: 'known' }, 1: { vals: F.h1, st: 'known', dead: F.z1.map(function (v) { return v <= 0; }) }, 2: { vals: dz2, st: 'grad', dead: F.z2.map(function (v) { return v <= 0; }) }, 3: { vals: dlogits, st: 'grad' } }, edges: { 1: { mode: 'bwd', mat: dW2, labels: true } } } },

    { phase: 'b', chip: '反向 6/7', title: 'dh¹ = (W²)ᵀ·dz²',
      desc: 'dz²₂ = 0 但 dh¹₂ ≠ 0：来自 W² 第 2 列的加权和。',
      formula: 'dh¹ = (W²)ᵀ · dz²', shape: '(4×4)·(4×1) → (4×1)',
      legend: '节点数值 = dh¹（红 = 梯度）',
      panel:
        ar('dh¹' + SB[0], [dot(W2[0][0], dz2[0]), dot(W2[1][0], dz2[1]), dot(W2[2][0], dz2[2]), dot(W2[3][0], dz2[3])], 0, fmt(dh1[0]), 'g') +
        ar('dh¹' + SB[1], [dot(W2[0][1], dz2[0]), dot(W2[1][1], dz2[1]), dot(W2[2][1], dz2[2]), dot(W2[3][1], dz2[3])], 0, fmt(dh1[1]), 'g') +
        ar('dh¹' + SB[2], [dot(W2[0][2], dz2[0]), dot(W2[1][2], dz2[1]), dot(W2[2][2], dz2[2]), dot(W2[3][2], dz2[3])], 0, fmt(dh1[2]), 'g') +
        ar('dh¹' + SB[3], [dot(W2[0][3], dz2[0]), dot(W2[1][3], dz2[1]), dot(W2[2][3], dz2[2]), dot(W2[3][3], dz2[3])], 0, fmt(dh1[3]), 'g'),
      g: { nodes: { 0: { vals: X, st: 'known' }, 1: { vals: dh1, st: 'grad' }, 2: { vals: dz2, st: 'grad', dead: F.z2.map(function (v) { return v <= 0; }) }, 3: { vals: dlogits, st: 'grad' } }, edges: { 1: { mode: 'bwd' } } } },

    { phase: 'b', chip: '反向 7/7', title: 'dz¹ = dh¹⊙1[z¹>0]，dW¹ = dz¹·xᵀ',
      desc: 'z¹₂ ≤ 0，dh¹₂ 归零；dW¹[i][j] = dz¹[i]·x[j]。',
      formula: 'dz¹ = dh¹ ⊙ 1[z¹>0]，dW¹ = dz¹·xᵀ，db¹ = dz¹', shape: '(4×1)⊙(4×1)，(4×1)·(1×2) → (4×2)',
      legend: '边上的红字 = dW¹　·　灰节点 = 梯度截断',
      panel:
        kv('dz¹' + SB[1] + '（截断后）', '0', 'r') +
        mat('dW¹（行 = z¹ 单元，列 = 输入 x₁, x₂）', ['1', '2', '3', '4'], dW1, '第 2 行全 0：dz¹₂ 被截断。db¹ = [' + dz1.map(function (v) { return fmt(v); }).join(', ') + ']。'),
      g: { nodes: { 0: { vals: X, st: 'known' }, 1: { vals: dz1, st: 'grad', dead: F.z1.map(function (v) { return v <= 0; }) }, 2: { vals: dz2, st: 'grad', dead: F.z2.map(function (v) { return v <= 0; }) }, 3: { vals: dlogits, st: 'grad' } }, edges: { 0: { mode: 'bwd', mat: dW1, labels: true } } } },

    { phase: 'u', chip: '更新 1/1', title: 'W ← W − η·dW（η = 0.3）',
      desc: '42 个参数更新后重跑前向：P₀ 翻过 0.5；两个截断单元仍为 0。',
      formula: 'W ← W − η·dW，b ← b − η·db', shape: '每个参数一步',
      legend: '节点数值 = 更新后重新前向的值（绿 = 已刷新）',
      panel:
        ar('W³₀₂', [fmt(W3[0][2], 1) + ' − ' + ETA + '·(' + fmt(dW3[0][2]) + ')'], null, fmt(W3[0][2] - ETA * dW3[0][2]), 'g') +
        ar('b³₀', [fmt(B3[0], 1) + ' − ' + ETA + '·(' + fmt(db3[0]) + ')'], null, fmt(B3[0] - ETA * db3[0]), 'g') +
        kv('参数总数', NPARAM + ' 个，全部同样更新') +
        kv('logits', fmt(F.lg[0]) + ' / ' + fmt(F.lg[1]) + ' → ' + fmt(G.lg[0]) + ' / ' + fmt(G.lg[1]), 'g') +
        kv('P₀（圆内）', fmtP(F.P[0]) + ' → ' + fmtP(G.P[0]), 'g') +
        kv('损失 L', fmt(F.L, 3) + ' → ' + fmt(G.L, 3), 'g'),
      g: { nodes: { 0: { vals: X, st: 'known' }, 1: { vals: G.h1, st: 'new', dead: G.z1.map(function (v) { return v <= 0; }) }, 2: { vals: G.h2, st: 'new', dead: G.z2.map(function (v) { return v <= 0; }) }, 3: { vals: G.P, st: 'new' } }, radius: { 3: [12 + 9 * G.P[0], 12 + 9 * G.P[1]] } } },
  ];

  var PHASE_LABEL = { f: '前向', l: '损失', b: '反向', u: '更新' };
  var PHASE_COUNT = { f: 6, l: 1, b: 7, u: 1 };
  var N = STEPS.length;

  /* ---------- SVG 布局 ---------- */
  var CX = [36, 128, 218, 300];                       // 4 列的 x
  var CY4 = [86, 148, 210, 272], CY2 = [141, 217];    // 每列节点的 y
  var COLS = [
    { cap: 'x', ys: CY2, names: ['x₁', 'x₂'], under: null },
    { cap: 'h¹ = ReLU(z¹)', ys: CY4, names: ['h¹₁', 'h¹₂', 'h¹₃', 'h¹₄'], under: null },
    { cap: 'h² = ReLU(z²)', ys: CY4, names: ['h²₁', 'h²₂', 'h²₃', 'h²₄'], under: null },
    { cap: 'logits → P', ys: CY2, names: ['ŷ₀', 'ŷ₁'], under: ['圆内 · 类 0', '圆外 · 类 1'] },
  ];
  var T4 = [0.26, 0.44, 0.62, 0.80], T2 = [0.42, 0.58];   // 边上标签的错位位置

  var SVGNS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) { e.setAttribute(k, attrs[k]); }
    if (parent) { parent.appendChild(e); }
    return e;
  }

  function mount(container) {
    var CSS = '' +
      '.bp-rail{display:flex;gap:8px;margin:12px 0 10px}' +
      '.bp-rail .bp-rg{display:flex;flex-direction:column;gap:3px;min-width:0}' +
      '.bp-rail .bp-rl{font:700 8.5px ' + MONO() + ';color:#65717c;letter-spacing:.06em;text-transform:uppercase;white-space:nowrap}' +
      '.bp-rail .bp-rs{display:flex;gap:2px}' +
      '.bp-rail .bp-rb{height:13px;flex:1 1 0;border:1px solid rgba(7,19,33,.45);background:#fffef9;cursor:pointer;padding:0;min-width:0}' +
      '.bp-rail .bp-rb.f.done{background:rgba(11,99,243,.38)}.bp-rail .bp-rb.f.cur{background:#0b63f3}' +
      '.bp-rail .bp-rb.l.done{background:rgba(7,19,33,.32)}.bp-rail .bp-rb.l.cur{background:#071321}' +
      '.bp-rail .bp-rb.b.done{background:rgba(192,57,43,.35)}.bp-rail .bp-rb.b.cur{background:#c0392b}' +
      '.bp-rail .bp-rb.u.done{background:rgba(200,255,71,.75)}.bp-rail .bp-rb.u.cur{background:#c8ff47}' +
      '.bp-main{display:flex;gap:12px;margin:6px 0 4px;align-items:flex-start}' +
      '.bp-graph{flex:0 0 300px;min-width:0}' +
      '.bp-graph svg{display:block;width:100%;height:auto;border:1px solid #d8d3c4;background:#fffef9}' +
      '.bp-graph text{font-family:' + MONO() + '}' +
      '.bp-cap{font-size:8.5px;font-weight:700;fill:#45515c}' +
      '.bp-nidx{font-size:8px;font-weight:700;fill:#65717c}' +
      '.bp-nval{font-size:9px;font-weight:800;fill:#071321}' +
      '.bp-nunder{font-size:7.5px;font-weight:700;fill:#65717c}' +
      '.bp-deadtag{font-size:7.5px;font-weight:800;fill:#c0392b}' +
      '.bp-elab{font-size:7.5px;font-weight:800;fill:#c0392b;paint-order:stroke;stroke:#fffef9;stroke-width:3px;stroke-linejoin:round}' +
      '.bp-nd circle{fill:#fffef9;stroke:rgba(7,19,33,.3);stroke-width:1;transition:r .3s ease}' +
      '.bp-nd.known circle{fill:#f4f2ec;stroke:#071321;stroke-width:1.4}' +
      '.bp-nd.new circle{fill:#c8ff47;stroke:#071321;stroke-width:2}' +
      '.bp-nd.dead circle{fill:#ebe8de;stroke:rgba(7,19,33,.4);stroke-dasharray:3 3}' +
      '.bp-nd.dead .bp-nval,.bp-nd.dead .bp-nidx{fill:#8b959e}' +
      '.bp-nd.grad circle{fill:#fdf2f0;stroke:#c0392b;stroke-width:1.6}' +
      '.bp-nd.grad .bp-nval{fill:#c0392b}' +
      '.bp-edge{stroke:rgba(7,19,33,.13);stroke-width:1}' +
      '.bp-edge.fwd{stroke:#0b63f3;stroke-width:1.6;stroke-dasharray:none}' +
      '.bp-edge.bwd{stroke:#c0392b;stroke-dasharray:5 3}' +
      '.bp-legend{font:700 8.5px ' + MONO() + ';color:#65717c;margin-top:5px;min-height:24px;line-height:1.5}' +
      '.bp-panel{flex:1 1 auto;min-width:0;border:1px solid rgba(7,19,33,.25);background:#fffef9;padding:10px 11px}' +
      '.bp-chip{display:inline-block;padding:2px 7px;border:1px solid #071321;font:800 8.5px ' + MONO() + ';letter-spacing:.08em;color:#071321;background:#f4f2ec;text-transform:uppercase}' +
      '.bp-chip.f{background:#0b63f3;color:#fffef9}.bp-chip.l{background:#071321;color:#fffef9}' +
      '.bp-chip.b{background:#c0392b;color:#fffef9}.bp-chip.u{background:#c8ff47;color:#071321}' +
      '.bp-h{margin:7px 0 4px;font:800 12.5px ' + MONO() + ';color:#071321}' +
      '.bp-desc{margin:0 0 8px;font-size:11px;line-height:1.55;color:#45515c}' +
      '.bp-line{font:700 9.5px ' + MONO() + ';color:#45515c;padding:3.5px 0;border-bottom:1px dashed rgba(7,19,33,.14);word-break:break-all}' +
      '.bp-line i{font-style:normal;color:#071321}' +
      '.bp-line b{color:#0b63f3}.bp-line b.g{color:#071321;background:rgba(200,255,71,.85);padding:0 3px}' +
      '.bp-kv{display:flex;justify-content:space-between;gap:8px;font:700 9.5px ' + MONO() + ';color:#45515c;padding:3.5px 0;border-bottom:1px dashed rgba(7,19,33,.14)}' +
      '.bp-kv b{color:#071321;white-space:nowrap}.bp-kv b.g{background:rgba(200,255,71,.85);padding:0 3px}' +
      '.bp-kv b.r{color:#c0392b}' +
      '.bp-mh{margin:6px 0 3px;font:700 8.5px ' + MONO() + ';color:#65717c;letter-spacing:.05em}' +
      '.bp-mat{display:grid;gap:2px}' +
      '.bp-ml{font:700 8px ' + MONO() + ';color:#65717c;display:grid;place-items:center;min-width:0}' +
      '.bp-mc{border:1px solid rgba(7,19,33,.35);background:#fffef9;font:700 8.5px ' + MONO() + ';color:#071321;text-align:center;padding:2.5px 0}' +
      '.bp-mc.z{color:#9aa4ad;border-style:dashed;border-color:rgba(7,19,33,.25)}' +
      '.bp-mnote{margin:4px 0 0;font-size:10px;color:#65717c;line-height:1.5}' +
      '.bp-formula{display:flex;justify-content:space-between;gap:10px;align-items:baseline;margin-top:10px;border:1px solid #071321;background:#f4f2ec;padding:8px 12px;flex-wrap:wrap}' +
      '.bp-formula b{font:800 12px ' + MONO() + ';color:#071321}' +
      '.bp-formula span{font:700 9px ' + MONO() + ';color:#65717c}' +
      '.wg-stat b.up{color:#0b63f3}' +
      '@media (max-width:620px){.bp-main{flex-direction:column}.bp-graph{flex:none}}';
    var style = document.createElement('style');
    style.textContent = CSS;
    container.appendChild(style);

    container.insertAdjacentHTML('beforeend',
      '<p class="wg-title">反向传播：一步一次走清楚</p>' +
      '<p class="wg-sub">固定 2−4−4−2 网络与样本，15 步拆开前向→损失→反向→更新。</p>' +
      '<div class="bp-rail" data-role="rail"></div>' +
      '<div class="bp-main">' +
        '<div class="bp-graph">' +
          '<svg data-role="svg" viewBox="0 0 320 304" role="img" aria-label="2-4-4-2 网络结构图"></svg>' +
          '<div class="bp-legend" data-role="legend"></div>' +
        '</div>' +
        '<div class="bp-panel" data-role="panel"></div>' +
      '</div>' +
      '<div class="wg-readout">' +
        '<div class="wg-stat"><span>当前步</span><b data-role="s-step">1 / 15</b></div>' +
        '<div class="wg-stat"><span>损失 L</span><b data-role="s-loss">—</b></div>' +
        '<div class="wg-stat"><span>P(圆内)</span><b data-role="s-p">—</b></div>' +
        '<div class="wg-stat"><span>学习率 η</span><b>0.3</b></div>' +
      '</div>' +
      '<div class="wg-controls">' +
        '<button type="button" class="wg-button" data-role="prev">← 上一步</button>' +
        '<button type="button" class="wg-button is-primary" data-role="next">下一步 →</button>' +
        '<button type="button" class="wg-button" data-role="auto">自动播放</button>' +
        '<button type="button" class="wg-button" data-role="reset">回到第 1 步</button>' +
      '</div>' +
      '<div class="bp-formula" data-role="formula"></div>' +
      '<p class="wg-note">面板显示保留 2−3 位小数。</p>');

    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };
    var svg = q('svg');

    /* 画网络 */
    COLS.forEach(function (col, c) {
      el('text', { x: CX[c], y: 16, 'text-anchor': 'middle', 'class': 'bp-cap' }, svg).textContent = col.cap;
    });
    var edgeLayer = [[], [], []];
    var edges = [];
    [0, 1, 2].forEach(function (layer) {
      var left = COLS[layer], right = COLS[layer + 1];
      for (var i = 0; i < left.ys.length; i++) {
        for (var j = 0; j < right.ys.length; j++) {
          var ln = el('line', { x1: CX[layer], y1: left.ys[i], x2: CX[layer + 1], y2: right.ys[j], 'class': 'bp-edge' }, svg);
          edges.push({ layer: layer, i: i, j: j, el: ln });
        }
      }
    });
    var elabels = el('g', {}, svg);
    var nodes = COLS.map(function (col, c) {
      return col.ys.map(function (y, i) {
        var g = el('g', { 'class': 'bp-nd' }, svg);
        var circle = el('circle', { cx: CX[c], cy: y, r: 15 }, g);
        var val = el('text', { x: CX[c], y: y + 3, 'text-anchor': 'middle', 'class': 'bp-nval' }, g);
        var idx = el('text', { x: CX[c], y: y - 20, 'text-anchor': 'middle', 'class': 'bp-nidx' }, g);
        idx.textContent = col.names[i];
        var under = null, dead = null;
        if (col.under) {
          under = el('text', { x: CX[c], y: y + 27, 'text-anchor': 'middle', 'class': 'bp-nunder' }, g);
          under.textContent = col.under[i];
        } else {
          dead = el('text', { x: CX[c], y: y + 27, 'text-anchor': 'middle', 'class': 'bp-deadtag', visibility: 'hidden' }, g);
          dead.textContent = '被截断';
        }
        return { g: g, circle: circle, val: val, dead: dead };
      });
    });

    /* 步骤条 */
    var rail = q('rail');
    var groups = {};
    var segEls = [];
    ['f', 'l', 'b', 'u'].forEach(function (ph) {
      var rg = document.createElement('div');
      rg.className = 'bp-rg';
      rg.style.flexGrow = String(PHASE_COUNT[ph]);
      var rl = document.createElement('span');
      rl.className = 'bp-rl';
      rl.textContent = PHASE_LABEL[ph] + ' ×' + PHASE_COUNT[ph];
      var rs = document.createElement('div');
      rs.className = 'bp-rs';
      rg.appendChild(rl); rg.appendChild(rs);
      rail.appendChild(rg);
      groups[ph] = rs;
    });
    STEPS.forEach(function (st, idx) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'bp-rb ' + st.phase;
      b.title = '第 ' + (idx + 1) + ' 步：' + st.title;
      b.addEventListener('click', function () { setStep(idx + 1); });
      groups[st.phase].appendChild(b);
      segEls.push(b);
    });

    var state = { step: 0, timer: null };
    var autoBtn = q('auto');

    function applyNodes(g) {
      var spec = g && g.nodes, radius = g && g.radius;
      nodes.forEach(function (colNodes, c) {
        var conf = spec && spec[c];
        colNodes.forEach(function (nd, i) {
          if (!conf) {
            nd.g.setAttribute('class', 'bp-nd');
            nd.val.textContent = '';
            nd.circle.setAttribute('r', 15);
            if (nd.dead) { nd.dead.setAttribute('visibility', 'hidden'); }
            return;
          }
          var dead = conf.dead && conf.dead[i];
          nd.g.setAttribute('class', 'bp-nd ' + (dead ? 'dead' : conf.st));
          var v = conf.vals[i];
          nd.val.textContent = fmt(v);
          nd.val.setAttribute('font-size', fmt(v).length > 4 ? '8' : '9');
          nd.circle.setAttribute('r', radius && radius[c] ? String(radius[c][i]) : '15');
          if (nd.dead) { nd.dead.setAttribute('visibility', dead ? 'visible' : 'hidden'); }
        });
      });
    }

    function applyEdges(spec) {
      edges.forEach(function (e) { e.el.setAttribute('class', 'bp-edge'); e.el.setAttribute('stroke-width', '1'); });
      while (elabels.firstChild) { elabels.removeChild(elabels.firstChild); }
      if (!spec) { return; }
      var layer = spec.layer, conf = spec.conf;
      var relevant = edges.filter(function (e) { return e.layer === layer; });
      var left = COLS[layer], right = COLS[layer + 1];
      var max = 1e-9, i, j;
      if (conf.mat) {
        for (i = 0; i < conf.mat.length; i++) { for (j = 0; j < conf.mat[i].length; j++) { max = Math.max(max, Math.abs(conf.mat[i][j])); } }
      }
      // 4×4 全连接太密：只标注 |梯度| 排前面的连接，完整数值看右侧矩阵
      var labMin = (conf.mat && left.ys.length === 4 && right.ys.length === 4) ? 0.45 * max : 0;
      relevant.forEach(function (e) {
        var g = conf.mat ? conf.mat[e.j][e.i] : 0;   // mat[目标][源]
        if (conf.mode === 'fwd') {
          e.el.setAttribute('class', 'bp-edge fwd');
        } else {
          e.el.setAttribute('class', 'bp-edge bwd');
          var w = 1 + 3.5 * Math.abs(g) / max;
          e.el.setAttribute('stroke-width', String(Math.min(4.5, w)));
        }
        if (conf.labels && conf.mat && Math.abs(g) > Math.max(0.0005, labMin)) {
          // 错位放在"只有 2 个节点"的那一侧，避免标签叠在一起
          var t = left.ys.length === 2 ? T2[e.i] : (right.ys.length === 2 ? T2[e.j] : T4[e.i]);
          var x = CX[layer] + t * (CX[layer + 1] - CX[layer]);
          var y = left.ys[e.i] + t * (right.ys[e.j] - left.ys[e.i]);
          var tx = el('text', { x: x, y: y + 2.5, 'text-anchor': 'middle', 'class': 'bp-elab' }, elabels);
          tx.textContent = fmt(g);
        }
      });
    }

    function setStep(n) {
      state.step = Math.max(1, Math.min(N, n));
      var st = STEPS[state.step - 1];
      segEls.forEach(function (b, idx) {
        b.className = 'bp-rb ' + STEPS[idx].phase + (idx + 1 < state.step ? ' done' : (idx + 1 === state.step ? ' cur' : ''));
      });
      applyNodes(st.g);
      applyEdges(st.g.edges ? { layer: Number(Object.keys(st.g.edges)[0]), conf: st.g.edges[Object.keys(st.g.edges)[0]] } : null);
      q('panel').innerHTML = '<span class="bp-chip ' + st.phase + '">' + st.chip + '</span>' +
        '<div class="bp-h">' + st.title + '</div><p class="bp-desc">' + st.desc + '</p>' + st.panel;
      q('legend').textContent = st.legend;
      q('formula').innerHTML = '<b>' + st.formula + '</b><span>' + st.shape + '</span>';
      q('s-step').textContent = state.step + ' / ' + N;
      var final = state.step === N;
      q('s-loss').textContent = final ? fmt(F.L, 3) + ' → ' + fmt(G.L, 3) : fmt(F.L, 3);
      q('s-loss').className = final ? 'up' : '';
      q('s-p').textContent = final ? fmtP(F.P[0]) + ' → ' + fmtP(G.P[0]) : fmtP(F.P[0]);
      q('s-p').className = final ? 'up' : '';
      q('prev').disabled = state.step === 1;
      q('next').disabled = final;
    }

    function stopAuto() {
      if (state.timer) { window.clearInterval(state.timer); state.timer = null; }
      autoBtn.textContent = '自动播放';
    }

    q('prev').addEventListener('click', function () { stopAuto(); setStep(state.step - 1); });
    q('next').addEventListener('click', function () { stopAuto(); setStep(state.step + 1); });
    q('reset').addEventListener('click', function () { stopAuto(); setStep(1); });
    autoBtn.addEventListener('click', function () {
      if (state.timer) { stopAuto(); return; }
      autoBtn.textContent = '暂停';
      state.timer = window.setInterval(function () {
        if (state.step >= N) { stopAuto(); return; }
        setStep(state.step + 1);
      }, 1600);
    });
    var onKey = function (ev) {
      if (ev.key === 'ArrowRight') { stopAuto(); setStep(state.step + 1); }
      else if (ev.key === 'ArrowLeft') { stopAuto(); setStep(state.step - 1); }
    };
    document.addEventListener('keydown', onKey);

    setStep(1);
    return function dispose() {
      stopAuto();
      document.removeEventListener('keydown', onKey);
    };
  }

  function MONO() { return 'ui-monospace, SFMono-Regular, Consolas, monospace'; }

  window.DLWidget.register('backprop-steps', mount);
})();
