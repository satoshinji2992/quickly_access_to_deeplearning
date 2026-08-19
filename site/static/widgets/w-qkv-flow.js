/* qkv-flow — QKV 分步：T=3、D=4 的迷你数字例子，从 X 一路算到 attention 输出。
 *
 * 六个阶段：① 输入 X ② X·Wq/Wk/Wv = Q/K/V ③ QKᵀ/√4 ④ +因果掩码
 * ⑤ 逐行 softmax ⑥ 权重·V。数值全部预计算（小整数，可手算核对），
 * 每步只新增本步的矩阵，过往矩阵保持可见可对照；第 ③ 步点击任意
 * score 格子，高亮参与计算的 Q 行与 K 行并展开点积公式。
 */
(function () {
  'use strict';

  var T = 3, D = 4, SQRTD = 2;   // √D = √4 = 2

  /* —— 预计算的全部数字 —— */
  var X = [
    [1, 0, 2, 0],
    [0, 1, 0, 2],
    [2, 1, 1, 0],
  ];
  var WQ = [
    [0, -1, 1, 0],
    [1, 0, 0, -1],
    [1, 0, -1, -1],
    [0, -1, -1, 0],
  ];
  var WK = [
    [0, -1, -1, 0],
    [0, 1, 0, -1],
    [-1, 0, 0, -1],
    [1, -1, 0, 1],
  ];
  var WV = [
    [-1, 1, 1, -1],
    [0, 0, -1, 1],
    [-1, 0, -1, 0],
    [-1, 0, 0, 1],
  ];

  function matmul(A, B) {
    var rows = A.length, cols = B[0].length, inner = B.length, out = [], i, j, k, s;
    for (i = 0; i < rows; i += 1) {
      out.push([]);
      for (j = 0; j < cols; j += 1) {
        s = 0;
        for (k = 0; k < inner; k += 1) { s += A[i][k] * B[k][j]; }
        out[i].push(s);
      }
    }
    return out;
  }

  var Q = matmul(X, WQ), K = matmul(X, WK), V = matmul(X, WV);
  function transpose(M) {
    var out = [], i, j;
    for (i = 0; i < M[0].length; i += 1) {
      out.push([]);
      for (j = 0; j < M.length; j += 1) { out[i].push(M[j][i]); }
    }
    return out;
  }
  var RAW = matmul(Q, transpose(K));          // QKᵀ，除以 √4 前的原始点积

  var SCORES = [], MASKED = [], WEIGHTS = [], OUT = [];
  (function precompute() {
    var i, j;
    for (i = 0; i < T; i += 1) {
      SCORES.push([]); MASKED.push([]); WEIGHTS.push([]);
      for (j = 0; j < T; j += 1) {
        SCORES[i].push(RAW[i][j] / SQRTD);
        MASKED[i].push(j > i ? -Infinity : RAW[i][j] / SQRTD);
        WEIGHTS[i].push(0);
      }
    }
    for (i = 0; i < T; i += 1) {
      var m = -Infinity;
      for (j = 0; j <= i; j += 1) { m = Math.max(m, MASKED[i][j]); }
      var sum = 0;
      for (j = 0; j <= i; j += 1) { sum += Math.exp(MASKED[i][j] - m); }
      for (j = 0; j <= i; j += 1) { WEIGHTS[i][j] = Math.exp(MASKED[i][j] - m) / sum; }
    }
    OUT = matmul(WEIGHTS, V);
  }());

  var fInt = function (v) { return String(v); };
  var f1 = function (v) { return (v < 0 ? '-' : '') + Math.abs(v).toFixed(1); };
  var f2 = function (v) { return (v < 0 ? '-' : '') + Math.abs(v).toFixed(2); };

  var MONO = 'ui-monospace, SFMono-Regular, Consolas, monospace';
  var CSS = '' +
    '.qf-steps{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0 10px}' +
    '.qf-chip{padding:5px 9px;border:1px solid rgba(7,19,33,.35);background:#f4f2ec;font:700 10px ' + MONO + ';color:#45515c;cursor:pointer;letter-spacing:.04em}' +
    '.qf-chip:hover{border-color:#071321;color:#071321}' +
    '.qf-chip.done{border-color:#0b63f3;color:#0b63f3}' +
    '.qf-chip.cur{background:#071321;border-color:#071321;color:#c8ff47}' +
    '.qf-stage{display:none;border:1px solid rgba(7,19,33,.28);background:#fffef9;padding:10px 12px;margin:10px 0}' +
    '.qf-stage.on{display:block;animation:qfin .38s ease}' +
    '.qf-stage.cur{border-color:#0b63f3;box-shadow:4px 4px 0 rgba(11,99,243,.16)}' +
    '@keyframes qfin{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}' +
    '.qf-slab{display:flex;justify-content:space-between;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:8px;font:800 11px ' + MONO + ';color:#071321;letter-spacing:.03em}' +
    '.qf-slab span{font:700 9px ' + MONO + ';color:#65717c;letter-spacing:.06em}' +
    '.qf-mat{display:grid;gap:2px;justify-content:start;width:max-content}' +
    '.qf-cc,.qf-chh,.qf-rh{font:700 8px ' + MONO + ';color:#8a94a0;height:12px;display:flex;align-items:flex-end;justify-content:center}' +
    '.qf-c{width:var(--cw);height:var(--ch);display:grid;place-items:center;border:1px solid rgba(7,19,33,.3);background:#fffef9;font:700 var(--fs,10px) ' + MONO + ';color:#071321}' +
    '.qf-c.neg{color:#c0392b}' +
    '.qf-c.mask{background:#e9e5d8;color:#97a0aa;border-color:rgba(7,19,33,.18)}' +
    '.qf-c.hlq{background:#d9e8ff;border-color:#0b63f3}' +
    '.qf-c.hlk{background:#d8f2ff;border-color:#2b9df0}' +
    '.qf-rh.hlq{color:#0b63f3;font-weight:800}' +
    '.qf-rh.hlk{color:#1290d6;font-weight:800}' +
    '.qf-sc{cursor:pointer;transition:background-color .15s}' +
    '.qf-sc:hover{border-color:#071321}' +
    '.qf-sc.rq{background:#e3edff}' +
    '.qf-sc.rk{background:#e6f6ff}' +
    '.qf-sc.sel{background:#c8ff47;box-shadow:inset 0 0 0 2px #071321}' +
    '.qf-s1{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}' +
    '.qf-side{flex:1 1 210px;font:700 10px/1.95 ' + MONO + ';color:#45515c}' +
    '.qf-side b{color:#071321}' +
    '.qf-proj{display:flex;gap:14px;flex-wrap:wrap}' +
    '.qf-pcol{flex:1 1 150px;min-width:150px}' +
    '.qf-ph{font:800 10px ' + MONO + ';padding:3px 7px;width:max-content;margin-bottom:6px;border:1px solid #071321}' +
    '.qf-ph.q{background:#0b63f3;color:#fffef9}' +
    '.qf-ph.k{background:#72d7ff;color:#071321}' +
    '.qf-ph.v{background:#071321;color:#fffef9}' +
    '.qf-parr{font:700 9px ' + MONO + ';color:#65717c;margin:5px 0}' +
    '.qf-ptag{margin-top:8px;font:700 9px ' + MONO + ';color:#45515c}' +
    '.qf-s3{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}' +
    '.qf-s3info{flex:1 1 240px;min-width:220px}' +
    '.qf-leg{display:flex;flex-direction:column;gap:5px;font:700 9px ' + MONO + ';color:#45515c;margin:2px 0 8px}' +
    '.qf-leg i{display:inline-block;width:14px;height:10px;border:1px solid rgba(7,19,33,.35);margin-right:6px;vertical-align:-1px}' +
    '.qf-form{border:1px dashed rgba(7,19,33,.35);background:#f7f5ef;padding:8px 10px;font:700 10px/1.9 ' + MONO + ';color:#263541;white-space:pre-wrap;word-break:break-all}' +
    '.qf-mrow{display:flex;align-items:flex-start;gap:10px;flex-wrap:wrap}' +
    '.qf-mitem{text-align:center}' +
    '.qf-mop{font:800 15px ' + MONO + ';color:#65717c;padding-top:26px}' +
    '.qf-mlab{font:700 9px ' + MONO + ';color:#65717c;margin-top:5px}' +
    '.qf-wrow{display:flex;gap:6px;align-items:flex-start}' +
    '.qf-sumcol{display:grid;gap:2px;margin-top:14px}' +
    '.qf-sumh{font:700 8px ' + MONO + ';color:#8a94a0;height:12px;display:flex;align-items:flex-end}' +
    '.qf-sumc{width:46px;height:32px;display:grid;place-items:center;border:1px dashed rgba(7,19,33,.3);background:#f4f2ec;font:700 10px ' + MONO + ';color:#65717c}' +
    '.qf-s6{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}' +
    '.qf-olines{flex:1 1 250px;font:700 10px/2.1 ' + MONO + ';color:#263541}' +
    '.qf-olines b{color:#0b63f3}' +
    '.qf-olines .qf-zero{color:#97a0aa}' +
    '.qf-ok0{margin-top:6px;border:1px solid rgba(7,19,33,.3);background:#f4f2ec;padding:6px 8px;font:700 9px/1.7 ' + MONO + ';color:#45515c}' +
    '.qf-ok0 b{color:#071321;background:#c8ff47;padding:0 3px}' +
    '.qf-controls .wg-button:disabled{opacity:.4;cursor:default}' +
    '.qf-controls .wg-button:disabled:hover{background:#fffef9;color:#071321;box-shadow:none}' +
    '.qf-controls .wg-button.is-primary:disabled{background:#071321;color:#c8ff47}' +
    '.qf-controls .wg-button.is-primary:disabled:hover{box-shadow:none}';

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined) { node.textContent = text; }
    return node;
  }

  /* data: 数值二维数组；opt: {rows, cols, cw, ch, fs, fmt, cellCls} */
  function buildMat(data, opt) {
    var wrap = el('div', 'qf-mat');
    wrap.style.setProperty('--cw', opt.cw + 'px');
    wrap.style.setProperty('--ch', opt.ch + 'px');
    wrap.style.setProperty('--fs', (opt.fs || 10) + 'px');
    var ncols = data[0].length;
    wrap.style.gridTemplateColumns = '16px repeat(' + ncols + ', var(--cw))';
    wrap.appendChild(el('i', 'qf-cc'));
    (opt.cols || []).forEach(function (c) { wrap.appendChild(el('i', 'qf-chh', c)); });
    var grid = [];
    data.forEach(function (row, i) {
      var lab = el('i', 'qf-rh', opt.rows ? opt.rows[i] : '');
      wrap.appendChild(lab);
      var rowCells = [];
      row.forEach(function (v, j) {
        var cell = el('b', 'qf-c' + (opt.cellCls ? ' ' + opt.cellCls : ''), opt.fmt(v, i, j));
        if (typeof v === 'number' && v < 0 && v !== -Infinity) { cell.classList.add('neg'); }
        wrap.appendChild(cell);
        rowCells.push(cell);
      });
      grid.push({ cells: rowCells, lab: lab });
    });
    return { root: wrap, grid: grid };
  }

  var ROW_T = ['t0', 't1', 't2'];
  var ROW_Q = ['q0', 'q1', 'q2'];
  var ROW_K = ['k0', 'k1', 'k2'];
  var ROW_V = ['v0', 'v1', 'v2'];
  var ROW_O = ['o0', 'o1', 'o2'];
  var COL_D = ['d0', 'd1', 'd2', 'd3'];

  var CHIP_LABELS = ['① X 输入', '② 三组投影', '③ QKᵀ 打分', '④ +M 掩码', '⑤ softmax', '⑥ ·V 输出'];
  var SLAB_TITLES = [
    '① 输入 X',
    '② 投影：同一输入、三组参数',
    '③ 打分 scores = QKᵀ/√4',
    '④ 因果掩码 scores + M',
    '⑤ 逐行 softmax',
    '⑥ 输出 out = 权重·V',
  ];
  var SLAB_SHAPES = [
    '(B,T,D) = (1,3,4)',
    'Wq/Wk/Wv (4,4) → Q/K/V (3,4)',
    '(T,T) = (3,3) · 点击格子',
    'j > i → −∞',
    '每行和 = 1',
    '(T,D) = (3,4)',
  ];
  var NOTES = [
    'X：3 个 token × 4 维；同一份 X 乘三组 W 得 Q/K/V',
    'Wq/Wk/Wv 各 (D,D)=(4,4)，投出三份不同的 (3,4)',
    'scores[i,j] = q_i·k_j / √4。点击格子高亮 Q 行与 K 列',
    'M 与 scores 同 shape：j ≤ i 加 0，j > i 加 −∞',
    '逐行 softmax：−∞ → 0，每行和 = 1；底色越深权重越大',
    'out = 权重·V；第 0 行权重 [1.00, 0, 0]，out₀ = v₀',
  ];

  function fmtIntCell(v) { return fInt(v); }
  function fmtScoreCell(v) { return f1(v); }
  function fmtWeightCell(v) { return f2(v); }

  function mount(container) {
    container.innerHTML = '';
    var style = document.createElement('style');
    style.textContent = CSS;
    container.appendChild(style);

    var shell = document.createElement('div');
    shell.innerHTML =
      '<p class="wg-title">QKV 分步：一个 attention 头的完整计算</p>' +
      '<p class="wg-sub">T=3、D=4。点「下一步」推进；第③步可点 scores 格子</p>' +
      '<div class="qf-steps" data-role="steps"></div>' +
      '<div class="wg-controls qf-controls">' +
        '<button type="button" class="wg-button" data-role="prev">上一步</button>' +
        '<button type="button" class="wg-button is-primary" data-role="next">下一步 →</button>' +
        '<button type="button" class="wg-button" data-role="auto">自动播放</button>' +
        '<button type="button" class="wg-button" data-role="reset">重置</button>' +
      '</div>' +
      '<div class="qf-stage" data-role="stage1"><div class="qf-slab"><b>' + SLAB_TITLES[0] + '</b><span>' + SLAB_SHAPES[0] + '</span></div><div class="qf-s1" data-role="s1body"></div></div>' +
      '<div class="qf-stage" data-role="stage2"><div class="qf-slab"><b>' + SLAB_TITLES[1] + '</b><span>' + SLAB_SHAPES[1] + '</span></div><div class="qf-proj" data-role="s2body"></div></div>' +
      '<div class="qf-stage" data-role="stage3"><div class="qf-slab"><b>' + SLAB_TITLES[2] + '</b><span>' + SLAB_SHAPES[2] + '</span></div><div class="qf-s3" data-role="s3body"></div></div>' +
      '<div class="qf-stage" data-role="stage4"><div class="qf-slab"><b>' + SLAB_TITLES[3] + '</b><span>' + SLAB_SHAPES[3] + '</span></div><div class="qf-mrow" data-role="s4body"></div></div>' +
      '<div class="qf-stage" data-role="stage5"><div class="qf-slab"><b>' + SLAB_TITLES[4] + '</b><span>' + SLAB_SHAPES[4] + '</span></div><div class="qf-wrow" data-role="s5body"></div></div>' +
      '<div class="qf-stage" data-role="stage6"><div class="qf-slab"><b>' + SLAB_TITLES[5] + '</b><span>' + SLAB_SHAPES[5] + '</span></div><div class="qf-s6" data-role="s6body"></div></div>' +
      '<p class="wg-note" data-role="note"></p>';
    container.appendChild(shell);

    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };
    var state = { step: 1, sel: [2, 1], timer: null };

    var initStep = parseInt(new URLSearchParams(window.location.search).get('qkv-step') || '1', 10);
    if (initStep >= 1 && initStep <= 6) { state.step = initStep; }

    /* —— ① X —— */
    var matX = buildMat(X, { rows: ROW_T, cols: COL_D, cw: 30, ch: 26, fs: 11, fmt: fmtIntCell });
    q('s1body').appendChild(matX.root);
    var side = el('div', 'qf-side');
    side.innerHTML =
      '<div><b>行 = token</b>：t0、t1、t2 共 T=3 个</div>' +
      '<div><b>列 = 维度</b>：d0–d3 共 D=4 维</div>' +
      '<div>例如 t2 = [2, 1, 1, 0]</div>' +
      '<div>带上 batch 即 <b>(B,T,D) = (1,3,4)</b></div>';
    q('s1body').appendChild(side);

    /* —— ② 投影 —— */
    var projRefs = [];
    [
      { name: 'Q', cls: 'q', W: WQ, M: Q, rows: ROW_Q },
      { name: 'K', cls: 'k', W: WK, M: K, rows: ROW_K },
      { name: 'V', cls: 'v', W: WV, M: V, rows: ROW_V },
    ].forEach(function (cfg) {
      var col = el('div', 'qf-pcol');
      col.appendChild(el('div', 'qf-ph ' + cfg.cls, cfg.name + ' = X·W' + cfg.name.toLowerCase()));
      var mW = buildMat(cfg.W, { rows: COL_D, cols: COL_D, cw: 26, ch: 23, fs: 10, fmt: fmtIntCell });
      col.appendChild(mW.root);
      col.appendChild(el('div', 'qf-parr', '同一个 X 乘 W' + cfg.name.toLowerCase() + ' ↓'));
      var mM = buildMat(cfg.M, { rows: cfg.rows, cols: COL_D, cw: 26, ch: 23, fs: 10, fmt: fmtIntCell });
      col.appendChild(mM.root);
      q('s2body').appendChild(col);
      projRefs.push({ mat: mM });
    });
    var ptag = el('div', 'qf-ptag', '三组 W 不同 → 同一个 X 投出三份 (3,4)');
    q('s2body').appendChild(ptag);
    var qRows = projRefs[0].mat.grid, kRows = projRefs[1].mat.grid;

    /* —— ③ scores —— */
    var matS = buildMat(SCORES, {
      rows: ROW_Q, cols: ROW_K, cw: 48, ch: 34, fs: 12, fmt: fmtScoreCell, cellCls: 'qf-sc',
    });
    q('s3body').appendChild(matS.root);
    var info = el('div', 'qf-s3info');
    var leg = el('div', 'qf-leg');
    leg.innerHTML =
      '<div><i style="background:#d9e8ff"></i>Q 第 i 行（scores 的行）</div>' +
      '<div><i style="background:#d8f2ff"></i>K 第 j 行（scores 的列）</div>' +
      '<div><i style="background:#c8ff47"></i>选中的格子 S[i,j]</div>';
    info.appendChild(leg);
    var form = el('div', 'qf-form');
    info.appendChild(form);
    q('s3body').appendChild(info);

    /* —— ④ 掩码 —— */
    var MASK = [];
    for (var mi = 0; mi < T; mi += 1) {
      MASK.push([]);
      for (var mj = 0; mj < T; mj += 1) { MASK[mi].push(mj > mi ? -Infinity : 0); }
    }
    function fmtMask(v) { return v === -Infinity ? '-∞' : '0'; }
    function fmtMasked(v) { return v === -Infinity ? '-∞' : f1(v); }
    var s4 = q('s4body');
    var mSc = buildMat(SCORES, { rows: ROW_Q, cols: ROW_K, cw: 36, ch: 30, fs: 10.5, fmt: fmtScoreCell });
    var mScWrap = el('div', 'qf-mitem'); mScWrap.appendChild(mSc.root);
    mScWrap.appendChild(el('div', 'qf-mlab', 'scores（第③步）'));
    s4.appendChild(mScWrap);
    s4.appendChild(el('div', 'qf-mop', '+'));
    var mM = buildMat(MASK, { rows: ROW_Q, cols: ROW_K, cw: 36, ch: 30, fs: 10.5, fmt: fmtMask, cellCls: 'mask' });
    var mMWrap = el('div', 'qf-mitem'); mMWrap.appendChild(mM.root);
    mMWrap.appendChild(el('div', 'qf-mlab', 'M 因果掩码'));
    s4.appendChild(mMWrap);
    s4.appendChild(el('div', 'qf-mop', '='));
    var mSm = buildMat(MASKED, { rows: ROW_Q, cols: ROW_K, cw: 38, ch: 30, fs: 10.5, fmt: fmtMasked });
    // buildMat 用 v<0 标红，认不出 -Infinity，这里手动补：上三角置灰。
    for (var si = 0; si < T; si += 1) {
      for (var sj = 0; sj < T; sj += 1) {
        if (sj > si) { mSm.grid[si].cells[sj].classList.add('mask'); }
      }
    }
    var mSmWrap = el('div', 'qf-mitem'); mSmWrap.appendChild(mSm.root);
    mSmWrap.appendChild(el('div', 'qf-mlab', 'scores + M'));
    s4.appendChild(mSmWrap);
    var s4note = el('div', 'qf-ptag', '灰色 = −∞：j > i 的位置被掩码');
    q('s4body').appendChild(s4note);

    /* —— ⑤ softmax —— */
    var matW = buildMat(WEIGHTS, { rows: ROW_Q, cols: ROW_K, cw: 48, ch: 32, fs: 11.5, fmt: fmtWeightCell });
    var wi, wj;
    for (wi = 0; wi < T; wi += 1) {
      for (wj = 0; wj < T; wj += 1) {
        var cell = matW.grid[wi].cells[wj], w = WEIGHTS[wi][wj];
        if (wj > wi) {
          cell.classList.add('mask');
        } else {
          var a = 0.07 + 0.9 * Math.pow(w, 0.85);
          cell.style.background = 'rgba(11,99,243,' + a.toFixed(3) + ')';
          if (w > 0.55) { cell.style.color = '#fffef9'; }
        }
      }
    }
    q('s5body').appendChild(matW.root);
    var sumcol = el('div', 'qf-sumcol');
    sumcol.appendChild(el('div', 'qf-sumh', 'Σ 行和'));
    for (wi = 0; wi < T; wi += 1) { sumcol.appendChild(el('div', 'qf-sumc', '1.00')); }
    q('s5body').appendChild(sumcol);
    var s5legend = el('div', 'qf-ptag');
    s5legend.style.cssText = 'flex:1 1 200px;margin-top:0';
    s5legend.innerHTML =
      '<div>底色越深 = 权重越大；灰格 = 0。</div>' +
      '<div>q0 只有 1 个可见位置 → 权重 1.00。</div>';
    q('s5body').appendChild(s5legend);

    /* —— ⑥ 输出 —— */
    var matO = buildMat(OUT, { rows: ROW_O, cols: COL_D, cw: 44, ch: 28, fs: 11, fmt: function (v) { return f2(v); } });
    q('s6body').appendChild(matO.root);
    var olines = el('div', 'qf-olines');
    var ohtml = '';
    for (wi = 0; wi < T; wi += 1) {
      ohtml += '<div>o' + wi + ' = ';
      var parts = [];
      for (wj = 0; wj <= wi; wj += 1) {
        var wv = WEIGHTS[wi][wj];
        parts.push('<b>' + f2(wv) + '</b>·v' + wj);
      }
      ohtml += parts.join(' + ') + '</div>';
    }
    olines.innerHTML = ohtml;
    var ok0 = el('div', 'qf-ok0');
    ok0.innerHTML = '<b>out₀ = v₀</b>　第一个 token 只看得到自己';
    olines.appendChild(ok0);
    q('s6body').appendChild(olines);

    /* —— 步骤条 —— */
    var chips = [];
    CHIP_LABELS.forEach(function (label, idx) {
      var chip = el('button', 'qf-chip', label);
      chip.type = 'button';
      chip.addEventListener('click', function () {
        stopAuto();
        state.step = idx + 1;
        render();
      });
      q('steps').appendChild(chip);
      chips.push(chip);
    });

    /* —— ③ 的点选高亮 —— */
    function fmtTerm(a, b) {
      var fa = a < 0 ? '(' + a + ')' : String(a);
      var fb = b < 0 ? '(' + b + ')' : String(b);
      return fa + '×' + fb;
    }
    function renderSel() {
      var i = state.sel[0], j = state.sel[1];
      qRows.forEach(function (r, ri) {
        r.lab.classList.toggle('hlq', ri === i);
        r.cells.forEach(function (c) { c.classList.toggle('hlq', ri === i); });
      });
      kRows.forEach(function (r, rj) {
        r.lab.classList.toggle('hlk', rj === j);
        r.cells.forEach(function (c) { c.classList.toggle('hlk', rj === j); });
      });
      matS.grid.forEach(function (row, ri) {
        row.lab.classList.toggle('hlq', ri === i);
        row.cells.forEach(function (c, cj) {
          c.classList.toggle('rq', ri === i && cj !== j);
          c.classList.toggle('rk', cj === j && ri !== i);
          c.classList.toggle('sel', ri === i && cj === j);
        });
      });
      var terms = [];
      for (var t = 0; t < D; t += 1) { terms.push(fmtTerm(Q[i][t], K[j][t])); }
      form.textContent =
        'q' + i + ' = [' + Q[i].join(', ') + ']    k' + j + ' = [' + K[j].join(', ') + ']\n' +
        'S[' + i + ',' + j + '] = q' + i + '·k' + j + ' / √4\n' +
        '     = (' + terms.join(' + ') + ') / 2\n' +
        '     = ' + RAW[i][j] + ' / 2 = ' + f1(SCORES[i][j]) +
        (j > i ? '   ← j>i，第④步会被掩码砍成 −∞' : '');
    }
    matS.grid.forEach(function (row, i) {
      row.cells.forEach(function (c, j) {
        c.addEventListener('click', function () {
          state.sel = [i, j];
          renderSel();
        });
      });
    });

    /* —— 总渲染 —— */
    var btnPrev = q('prev'), btnNext = q('next'), btnAuto = q('auto'), btnReset = q('reset');
    function render() {
      var st;
      for (st = 1; st <= 6; st += 1) {
        var node = q('stage' + st);
        node.classList.toggle('on', st <= state.step);
        node.classList.toggle('cur', st === state.step);
      }
      chips.forEach(function (chip, idx) {
        chip.classList.toggle('cur', idx + 1 === state.step);
        chip.classList.toggle('done', idx + 1 < state.step);
      });
      btnPrev.disabled = state.step <= 1;
      btnNext.disabled = state.step >= 6;
      btnNext.textContent = state.step >= 6 ? '已完成' : '下一步 →';
      q('note').textContent = '第 ' + state.step + ' / 6 步 · ' + NOTES[state.step - 1];
      renderSel();
    }

    function stopAuto() {
      if (state.timer) {
        window.clearInterval(state.timer);
        state.timer = null;
        btnAuto.textContent = '自动播放';
      }
    }
    btnPrev.addEventListener('click', function () { stopAuto(); if (state.step > 1) { state.step -= 1; render(); } });
    btnNext.addEventListener('click', function () { stopAuto(); if (state.step < 6) { state.step += 1; render(); } });
    btnReset.addEventListener('click', function () {
      stopAuto();
      state.step = 1; state.sel = [2, 1];
      render();
    });
    btnAuto.addEventListener('click', function () {
      if (state.timer) { stopAuto(); return; }
      if (state.step >= 6) { state.step = 1; }
      btnAuto.textContent = '暂停';
      state.timer = window.setInterval(function () {
        if (state.step >= 6) { stopAuto(); return; }
        state.step += 1;
        render();
      }, 1500);
      render();
    });

    render();
    return function dispose() { stopAuto(); };
  }

  window.DLWidget.register('qkv-flow', mount);
})();
