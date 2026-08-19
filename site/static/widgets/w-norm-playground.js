/* norm-playground — 4×4 数据矩阵上对比 BN（逐列跨样本）与 LN（逐行跨特征），点击改值实时重算，BN 侧带 EMA running 统计条。 */
(function () {
  'use strict';

  var INK = '#071321', BLUE = '#0b63f3', RED = '#c0392b', PAPER = '#f7f5ef', GREEN = '#c8ff47';
  var MONO = 'ui-monospace,SFMono-Regular,Consolas,monospace';

  // 初始 batch：4 列均值差异大（约 +6.3 / −3.3 / +1.3 / +0.3），BN 按列、LN 按行的统计完全不同。
  var X0 = [
    [6.5, -3.5, 1.0, 0.5],
    [5.5, -2.5, 2.0, -1.5],
    [7.0, -4.0, 1.5, 2.5],
    [6.0, -3.0, 0.5, -0.5],
  ];
  // 「再来一批」的取样分布：每个特征有固定的真实均值与波动。
  var COL_MU = [6, -3, 1.2, 0];
  var COL_SD = [0.9, 0.8, 0.7, 1.5];
  var EPS2 = 1e-4;  // σ² 下限，防除零
  var MOM = 0.7;    // running 统计的 EMA momentum

  var CSS = ''
    + '.np-grid{display:grid;gap:3px;user-select:none;grid-template-columns:36px repeat(4,minmax(0,1fr)) 10px repeat(4,minmax(0,1fr)) 10px repeat(4,minmax(0,1fr)) 52px;grid-template-rows:26px 16px repeat(4,auto) 13px 13px;}'
    + '.np-side,.np-flab{display:grid;place-items:center;text-align:center;font:700 8.5px ' + MONO + ';letter-spacing:.03em;color:#65717c;line-height:1.25;}'
    + '.np-phead{display:flex;align-items:center;justify-content:center;padding:3px 4px;border:1px solid #071321;font:800 9.5px ' + MONO + ';letter-spacing:.02em;white-space:nowrap;overflow:hidden;}'
    + '.np-phead.is-raw{background:#fffef9;color:#071321;}'
    + '.np-phead.is-bn{background:#0b63f3;color:#fffef9;}'
    + '.np-phead.is-ln{background:#071321;color:#fffef9;}'
    + '.np-cell{display:grid;place-items:center;aspect-ratio:1;min-width:0;border:1px solid rgba(7,19,33,.3);background:#fffef9;font:700 10px ' + MONO + ';color:#071321;overflow:hidden;transition:box-shadow .18s ease,border-color .18s ease;}'
    + '@media(max-width:640px){.np-grid{grid-template-columns:24px repeat(4,minmax(0,1fr)) 8px repeat(4,minmax(0,1fr)) 8px repeat(4,minmax(0,1fr)) 38px;gap:2px}.np-cell{font-size:8.5px}.np-phead{font-size:8px;padding:2px}.np-side,.np-flab{font-size:7.5px}}'
    + '.np-cell.np-edit{cursor:pointer;}'
    + '.np-cell.np-edit:hover{border-color:#0b63f3;box-shadow:0 0 0 2px rgba(11,99,243,.35);}'
    + '.np-fstat{display:grid;place-items:center;font:700 8.5px ' + MONO + ';color:#45515c;}'
    + '.np-lstat{display:grid;place-content:center;justify-items:center;gap:2px;border:1px dashed rgba(7,19,33,.3);background:#f7f5ef;font:700 8.5px ' + MONO + ';color:#45515c;line-height:1.1;}'
    + '.np-new{box-shadow:0 0 0 2px #c8ff47 !important;border-color:#071321 !important;}'
    + '.np-hit{box-shadow:0 0 0 2px rgba(11,99,243,.65);border-color:#0b63f3;}'
    + '.np-run{display:grid;grid-template-columns:44px minmax(0,1fr) 56px minmax(0,1fr) 56px;gap:7px 8px;align-items:center;margin:10px 0 4px;}'
    + '.np-runhead{font:700 8.5px ' + MONO + ';letter-spacing:.05em;color:#65717c;text-transform:uppercase;white-space:nowrap;overflow:hidden;}'
    + '.np-rlab{font:700 10px ' + MONO + ';color:#071321;}'
    + '.np-track{position:relative;height:10px;border:1px solid rgba(7,19,33,.5);background:#fffef9;}'
    + '.np-mid{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:rgba(7,19,33,.4);}'
    + '.np-fill{position:absolute;top:0;bottom:0;transition:left .25s ease,width .25s ease;}'
    + '.np-tick{position:absolute;top:-3px;bottom:-3px;width:3px;box-sizing:border-box;background:#c8ff47;border:1px solid #071321;transition:left .25s ease;}'
    + '.np-tick.np-pulse{top:-5px;bottom:-5px;}'
    + '.np-val{font:700 10px ' + MONO + ';color:#071321;text-align:right;}';

  function clone(m) { return m.map(function (r) { return r.slice(); }); }

  function colStats(X) {  // 每列跨 4 个样本：BN 的统计量
    var mu = [], vr = [];
    for (var c = 0; c < 4; c++) {
      var m = 0;
      for (var b = 0; b < 4; b++) { m += X[b][c]; }
      m /= 4;
      var s = 0;
      for (var b2 = 0; b2 < 4; b2++) { var d = X[b2][c] - m; s += d * d; }
      mu.push(m); vr.push(s / 4);
    }
    return { mu: mu, vr: vr };
  }

  function rowStats(X) {  // 每行跨 4 个特征：LN 的统计量
    var mu = [], vr = [];
    for (var b = 0; b < 4; b++) {
      var m = 0;
      for (var c = 0; c < 4; c++) { m += X[b][c]; }
      m /= 4;
      var s = 0;
      for (var c2 = 0; c2 < 4; c2++) { var d = X[b][c2] - m; s += d * d; }
      mu.push(m); vr.push(s / 4);
    }
    return { mu: mu, vr: vr };
  }

  function gauss() {  // Box-Muller
    var u = Math.random() || 1e-9, v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function round1(v) { return Math.round(v * 10) / 10; }

  // 发散配色：正红、负蓝，越深绝对值越大（scale 固定，颜色可跨视图比较）。
  function divCol(v, scale) {
    var t = Math.max(0, Math.min(1, Math.abs(v) / scale));
    if (v > 1e-9) { return { bg: 'rgba(192,57,43,' + (0.08 + 0.7 * t).toFixed(3) + ')', fg: t > 0.55 ? '#fffef9' : INK }; }
    if (v < -1e-9) { return { bg: 'rgba(11,99,243,' + (0.08 + 0.7 * t).toFixed(3) + ')', fg: t > 0.55 ? '#fffef9' : INK }; }
    return { bg: PAPER, fg: '#65717c' };
  }

  function mount(container) {
    var styleEl = document.createElement('style');
    styleEl.textContent = CSS;
    container.appendChild(styleEl);

    var root = document.createElement('div');
    root.innerHTML =
      '<p class="wg-title">BatchNorm 与 LayerNorm：沿不同的方向做标准化</p>' +
      '<p class="wg-sub">同一份 4 样本 × 4 特征的数据 X: (B=4, D=4)。BN 逐<b>列</b>跨 4 个样本统计 μc、σc（依赖 batch）；LN 逐<b>行</b>跨 4 个特征统计 μr、σr（与 batch 无关）。点击「原始值」的格子 +0.5（Shift 点击 −0.5），右边两个视图实时重算。</p>' +
      '<div class="np-grid" data-role="grid"></div>' +
      '<p class="wg-note">配色：红 = 正值、蓝 = 负值，颜色越深绝对值越大。归一化后 BN 的每列、LN 的每行都满足 μ≈0、σ≈1（分母 σ 加了 ε²=1e-4 防除零，真实实现里 ε≈1e-5）。</p>' +
      '<div class="wg-block">' +
        '<div class="wg-label"><span>BN 的 running 统计（推理时用）</span><span data-role="nb"></span></div>' +
        '<div class="wg-controls">' +
          '<button type="button" class="wg-button is-primary" data-role="new">再来一批</button>' +
          '<button type="button" class="wg-button" data-role="reset">重置数据</button>' +
        '</div>' +
        '<div class="np-run" data-role="run"></div>' +
        '<p class="wg-note">实条 = running 统计（EMA 累积），绿标 = 本批统计。每按一次「再来一批」：X 换成 4 个新样本，running ← 0.7·running + 0.3·本批。点击格子改值只影响当前视图，不动 running。</p>' +
        '<p class="wg-note" data-role="hint"></p>' +
        '<p class="wg-note">要点：BN 的统计<b>跨样本</b>（同一列的 4 个数），换一批样本结果就变，所以推理时要用 running 统计兜底；LN 的统计<b>跨特征</b>（同一行的 4 个数），单个样本自己就能算，与 batch 无关——这正是 Transformer 用 LN 的原因。</p>' +
      '</div>';
    container.appendChild(root);

    var q = function (role) { return root.querySelector('[data-role="' + role + '"]'); };
    var timers = [];
    function later(fn, ms) { timers.push(window.setTimeout(fn, ms)); }

    var X = clone(X0);
    var cs0 = colStats(X);
    var state = { runMu: cs0.mu.slice(), runVr: cs0.vr.slice(), batches: 1 };

    // ---- 主网格：三块矩阵共享一套行列，保证行对齐 ----
    var grid = q('grid');
    function place(el, col, row, span) {
      el.style.gridColumn = span && span > 1 ? (col + ' / span ' + span) : String(col);
      el.style.gridRow = String(row);
      grid.appendChild(el);
      return el;
    }
    function div(cls, text) {
      var el = document.createElement('div');
      el.className = cls;
      if (text !== undefined) { el.textContent = text; }
      return el;
    }

    place(div('np-side', '样本 ↓'), 1, 1);
    place(div('np-phead is-raw', '原始值 X · 点击 +0.5'), 2, 1, 4);
    place(div('np-phead is-bn', 'BN · 每列 μc σc'), 7, 1, 4);
    place(div('np-phead is-ln', 'LN · 每行 μr σr'), 12, 1, 4);
    place(div('np-side', '特征 →'), 1, 2);
    var c1, c2;
    for (c1 = 0; c1 < 4; c1++) {
      place(div('np-flab', '特征' + (c1 + 1)), 2 + c1, 2);
      place(div('np-flab', '特征' + (c1 + 1)), 7 + c1, 2);
      place(div('np-flab', '特征' + (c1 + 1)), 12 + c1, 2);
    }
    place(div('np-side', '行统计'), 16, 2);

    var rawCells = [], bnCells = [], lnCells = [], bnMuEls = [], bnSdEls = [], lnStatEls = [];
    var b, c;
    for (b = 0; b < 4; b++) {
      place(div('np-side', '样本' + (b + 1)), 1, 3 + b);
      for (c = 0; c < 4; c++) {
        (function (rb, rc) {
          var cell = div('np-cell np-edit');
          cell.title = '点击 +0.5 · Shift 点击 −0.5';
          cell.setAttribute('role', 'button');
          cell.addEventListener('click', function (e) {
            var delta = e.shiftKey ? -0.5 : 0.5;
            var next = Math.max(-12, Math.min(12, round1(X[rb][rc] + delta)));
            if (next === X[rb][rc]) { return; }
            X[rb][rc] = next;
            render();
            pulse(rawCells[rb * 4 + rc], bnCells, lnCells, rb, rc);
            q('hint').textContent = '样本' + (rb + 1) + '·特征' + (rc + 1) + ' → ' + next.toFixed(1) +
              '：该列 μc 和该行 μr 都移动 0.125，BN 重排整列（蓝框），LN 重排整行（蓝框）。';
          });
          place(cell, 2 + rc, 3 + rb);
          rawCells.push(cell);
        }(b, c));
        bnCells.push(place(div('np-cell'), 7 + c, 3 + b));
        lnCells.push(place(div('np-cell'), 12 + c, 3 + b));
      }
      var ls = div('np-lstat');
      var l1 = div('', 'μ —'), l2 = div('', 'σ —');
      ls.appendChild(l1); ls.appendChild(l2);
      place(ls, 16, 3 + b);
      lnStatEls.push({ mu: l1, sd: l2 });
    }
    for (c2 = 0; c2 < 4; c2++) {
      var fm = place(div('np-fstat'), 7 + c2, 7);
      var fs = place(div('np-fstat'), 7 + c2, 8);
      bnMuEls.push(fm); bnSdEls.push(fs);
    }

    // 点击后：被改的格子绿框，BN 整列 / LN 整行蓝框，短暂高亮。
    function pulse(rawEl, bnAll, lnAll, rb, rc) {
      rawEl.classList.add('np-new');
      later(function () { rawEl.classList.remove('np-new'); }, 500);
      for (var i = 0; i < 4; i++) {
        (function (bc, lr) {
          bc.classList.add('np-hit'); lr.classList.add('np-hit');
          later(function () { bc.classList.remove('np-hit'); lr.classList.remove('np-hit'); }, 500);
        }(bnAll[i * 4 + rc], lnAll[rb * 4 + i]));
      }
    }

    // ---- running 统计条 ----
    var run = q('run');
    var MU_RANGE = 8, SD_RANGE = 3;
    function track() {
      var el = document.createElement('div');
      el.className = 'np-track';
      return el;
    }
    var runRows = [];
    var head = [
      div('np-runhead', ''), div('np-runhead', 'running μ · 绿标 = 本批 μc'), div('np-runhead', ''),
      div('np-runhead', 'running σ · 绿标 = 本批 σc'), div('np-runhead', ''),
    ];
    head.forEach(function (h) { run.appendChild(h); });
    for (var f = 0; f < 4; f++) {
      var lab = div('np-rlab', '特征' + (f + 1));
      var muT = track(), muF = document.createElement('div'), muTick = document.createElement('div');
      muF.className = 'np-fill'; muTick.className = 'np-tick';
      muT.appendChild(div('np-mid')); muT.appendChild(muF); muT.appendChild(muTick);
      var sdT = track(), sdF = document.createElement('div'), sdTick = document.createElement('div');
      sdF.className = 'np-fill'; sdTick.className = 'np-tick';
      sdT.appendChild(sdF); sdT.appendChild(sdTick);
      var muV = div('np-val'), sdV = div('np-val');
      [lab, muT, muV, sdT, sdV].forEach(function (el) { run.appendChild(el); });
      runRows.push({ muF: muF, muT: muTick, muV: muV, sdF: sdF, sdT: sdTick, sdV: sdV });
    }

    function paintBar(fill, tick, valEl, v, batchV, kind) {
      var fillPct, tickPct;
      if (kind === 'mu') {  // 发散：中心 0，正右负左
        var p = Math.max(-MU_RANGE, Math.min(MU_RANGE, v)) / MU_RANGE;
        var bp = Math.max(-MU_RANGE, Math.min(MU_RANGE, batchV)) / MU_RANGE;
        fillPct = p >= 0 ? { left: '50%', width: (p * 50) + '%' } : { left: (50 + p * 50) + '%', width: (-p * 50) + '%' };
        tickPct = (50 + bp * 50);
        fill.style.background = v >= 0 ? 'rgba(192,57,43,.75)' : 'rgba(11,99,243,.75)';
      } else {              // σ 只有正值
        var w = Math.max(0, Math.min(SD_RANGE, v)) / SD_RANGE;
        var bw = Math.max(0, Math.min(SD_RANGE, batchV)) / SD_RANGE;
        fillPct = { left: '0%', width: (w * 100) + '%' };
        tickPct = bw * 100;
        fill.style.background = 'rgba(7,19,33,.7)';
      }
      fill.style.left = fillPct.left; fill.style.width = fillPct.width;
      tick.style.left = 'calc(' + tickPct.toFixed(2) + '% - 1.5px)';
      valEl.textContent = v.toFixed(2);
    }

    function render() {
      var cs = colStats(X), rs = rowStats(X);
      var n, bb, cc;
      for (bb = 0; bb < 4; bb++) {
        for (cc = 0; cc < 4; cc++) {
          var i = bb * 4 + cc;
          var rv = X[bb][cc];
          rawCells[i].textContent = rv.toFixed(1);
          var rc = divCol(rv, 8);
          rawCells[i].style.background = rc.bg; rawCells[i].style.color = rc.fg;
          var bv = (rv - cs.mu[cc]) / Math.sqrt(cs.vr[cc] + EPS2);
          bnCells[i].textContent = bv.toFixed(2);
          var bc = divCol(bv, 2);
          bnCells[i].style.background = bc.bg; bnCells[i].style.color = bc.fg;
          var lv = (rv - rs.mu[bb]) / Math.sqrt(rs.vr[bb] + EPS2);
          lnCells[i].textContent = lv.toFixed(2);
          var lc = divCol(lv, 2);
          lnCells[i].style.background = lc.bg; lnCells[i].style.color = lc.fg;
        }
        lnStatEls[bb].mu.textContent = 'μ ' + rs.mu[bb].toFixed(2);
        lnStatEls[bb].sd.textContent = 'σ ' + Math.sqrt(rs.vr[bb]).toFixed(2);
      }
      for (n = 0; n < 4; n++) {
        bnMuEls[n].textContent = 'μ ' + cs.mu[n].toFixed(2);
        bnSdEls[n].textContent = 'σ ' + Math.sqrt(cs.vr[n]).toFixed(2);
      }
      for (var r = 0; r < 4; r++) {
        paintBar(runRows[r].muF, runRows[r].muT, runRows[r].muV, state.runMu[r], cs.mu[r], 'mu');
        paintBar(runRows[r].sdF, runRows[r].sdT, runRows[r].sdV, Math.sqrt(state.runVr[r]), Math.sqrt(cs.vr[r]), 'sd');
      }
      q('nb').textContent = '已见 ' + state.batches + ' 批 · momentum = 0.7';
    }

    function flashTicks() {
      runRows.forEach(function (row) {
        row.muT.classList.add('np-pulse'); row.sdT.classList.add('np-pulse');
        later(function () { row.muT.classList.remove('np-pulse'); row.sdT.classList.remove('np-pulse'); }, 450);
      });
    }

    q('new').addEventListener('click', function () {
      for (var b2 = 0; b2 < 4; b2++) {
        for (var c3 = 0; c3 < 4; c3++) { X[b2][c3] = round1(COL_MU[c3] + COL_SD[c3] * gauss()); }
      }
      var cs = colStats(X);
      for (var k = 0; k < 4; k++) {
        state.runMu[k] = MOM * state.runMu[k] + (1 - MOM) * cs.mu[k];
        state.runVr[k] = MOM * state.runVr[k] + (1 - MOM) * cs.vr[k];
      }
      state.batches += 1;
      render();
      flashTicks();
      q('hint').textContent = '第 ' + state.batches + ' 批：X 换成 4 个新样本——BN 视图随 batch 成员变化，LN 的每行仍只用自身统计。running 向本批靠近 30%。';
    });

    q('reset').addEventListener('click', function () {
      X = clone(X0);
      var cs = colStats(X);
      state.runMu = cs.mu.slice();
      state.runVr = cs.vr.slice();
      state.batches = 1;
      render();
      q('hint').textContent = '已重置回初始批：running 统计以第 1 批为初值。';
    });

    render();
    q('hint').textContent = '初始批：4 列均值差异很大（μc ≈ +6.25 / −3.25 / +1.25 / +0.25）。同一份数据，BN 按列、LN 按行，归一化结果完全不同。';

    return function dispose() {
      timers.forEach(function (t) { window.clearTimeout(t); });
      timers = [];
      if (styleEl.parentNode) { styleEl.parentNode.removeChild(styleEl); }
    };
  }

  window.DLWidget.register('norm-playground', mount);
})();
