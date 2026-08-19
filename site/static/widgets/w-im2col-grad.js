/* im2col-grad — 4×4 输入 / 3×3 核 / C_out=2：前向点开 Y_col 看乘积来源，反向看 dW = X_colᵀ·dY 与 col2im 累加。 */
(function () {
  'use strict';

  var INK = '#071321', CARD = '#fffef9', PAPER = '#f7f5ef';

  // 4×4 输入：值域 0…3，便于手算。
  var X = [
    [1, 2, 0, 1],
    [0, 1, 3, 2],
    [2, 3, 1, 0],
    [1, 2, 2, 1],
  ];

  // 4 个窗口，行优先编号 0..3，对应 X_col 的 4 行。
  var WIN = [[0, 0], [0, 1], [1, 0], [1, 1]];

  // W_col (9×2)：第 k 行 = 3×3 核内行优先第 k 个 tap 在两个输出通道的权重。
  var WCOL = [
    [1, 0], [0, 1], [-1, 0],
    [0, 1], [1, -2], [0, 1],
    [-1, 0], [0, 1], [1, 0],
  ];

  // 上游梯度 dY_col (4×2) 预设。
  var DY = [
    [1, 0],
    [0, 1],
    [-1, 1],
    [2, -1],
  ];

  // ---- 派生矩阵（全部整数，可手算验证） ----
  var XCOL = [], YCOL = [], DW = [], DXCOL = [], DX = [];
  var r, c, k, o, i, j;

  for (r = 0; r < 4; r++) {
    XCOL[r] = [];
    for (k = 0; k < 9; k++) {
      XCOL[r][k] = X[WIN[r][0] + Math.floor(k / 3)][WIN[r][1] + (k % 3)];
    }
  }
  for (r = 0; r < 4; r++) {
    YCOL[r] = [0, 0];
    for (o = 0; o < 2; o++) {
      for (k = 0; k < 9; k++) { YCOL[r][o] += XCOL[r][k] * WCOL[k][o]; }
    }
  }
  for (k = 0; k < 9; k++) {
    DW[k] = [0, 0];
    for (o = 0; o < 2; o++) {
      for (r = 0; r < 4; r++) { DW[k][o] += XCOL[r][k] * DY[r][o]; }
    }
  }
  for (r = 0; r < 4; r++) {
    DXCOL[r] = [];
    for (k = 0; k < 9; k++) {
      DXCOL[r][k] = 0;
      for (o = 0; o < 2; o++) { DXCOL[r][k] += WCOL[k][o] * DY[r][o]; }
    }
  }
  for (i = 0; i < 4; i++) {
    DX[i] = [];
    for (j = 0; j < 4; j++) { DX[i][j] = 0; }
  }
  for (r = 0; r < 4; r++) {
    for (k = 0; k < 9; k++) {
      DX[WIN[r][0] + Math.floor(k / 3)][WIN[r][1] + (k % 3)] += DXCOL[r][k];
    }
  }

  // 覆盖像素 (i,j) 的窗口：返回 [{ r, k }, ...]，k = 该像素在此窗口行的 tap 位。
  function windowsOf(pi, pj) {
    var out = [];
    for (var n = 0; n < 4; n++) {
      var wr = WIN[n][0], wc = WIN[n][1];
      if (pi >= wr && pi <= wr + 2 && pj >= wc && pj <= wc + 2) {
        out.push({ r: n, k: (pi - wr) * 3 + (pj - wc) });
      }
    }
    return out;
  }

  function inputColor(v) {
    var a = v === 0 ? 0.05 : 0.10 + 0.30 * (v / 3);
    return 'rgba(11,99,243,' + a.toFixed(3) + ')';
  }

  // 发散配色：正值偏红、负值偏蓝、零为纸色。
  function divergent(v, scale) {
    var t = Math.max(0, Math.min(1, Math.abs(v) / (scale || 1)));
    if (v > 0) { return { bg: 'rgba(192,57,43,' + (0.10 + 0.60 * t).toFixed(3) + ')', fg: t > 0.62 ? CARD : INK }; }
    if (v < 0) { return { bg: 'rgba(11,99,243,' + (0.10 + 0.60 * t).toFixed(3) + ')', fg: t > 0.62 ? CARD : INK }; }
    return { bg: PAPER, fg: '#8b949c' };
  }

  function fmt(v) {
    return String(v).replace('-', '−');
  }

  var CSS = ''
    + '.ig-shape{margin:2px 0 10px;font:700 10px ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.06em;color:#45515c;}'
    + '.ig-top{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;}'
    + '.ig-gcol{display:flex;flex-direction:column;min-width:0;}'
    + '.ig-fixed{position:relative;width:max-content;}'
    + '.ig-xgrid{display:grid;grid-template-columns:repeat(4,34px);gap:3px;}'
    + '.ig-xcell{width:34px;height:34px;display:grid;place-items:center;box-sizing:border-box;border:1px solid rgba(7,19,33,.3);background:#fffef9;font:700 12px ui-monospace,SFMono-Regular,Consolas,monospace;color:#071321;transition:opacity .2s,box-shadow .2s,border-color .2s;}'
    + '.ig-xcell.ig-click{cursor:pointer;}'
    + '.ig-xcell.ig-click:hover{border-color:#071321;}'
    + '.ig-frame{position:absolute;box-sizing:border-box;border:2px solid #0b63f3;box-shadow:0 0 0 3px rgba(11,99,243,.16);pointer-events:none;z-index:2;transition:left .18s ease,top .18s ease,width .18s ease,height .18s ease;}'
    + '.ig-fbadge{position:absolute;top:-9px;left:-2px;background:#0b63f3;color:#fffef9;padding:0 3px;font:800 8px ui-monospace,SFMono-Regular,Consolas,monospace;line-height:1.5;white-space:nowrap;}'
    + '.ig-kboxes{display:flex;gap:10px;}'
    + '.ig-kbox{padding:5px 6px 7px;border:1px solid rgba(7,19,33,.3);background:#fffef9;transition:box-shadow .2s,border-color .2s;}'
    + '.ig-kbox.is-on{border-color:#0b63f3;box-shadow:0 0 0 3px rgba(11,99,243,.22);}'
    + '.ig-kgrid{display:grid;grid-template-columns:repeat(3,26px);gap:3px;margin-top:4px;}'
    + '.ig-kcell{width:26px;height:22px;display:grid;place-items:center;box-sizing:border-box;border:1px solid rgba(7,19,33,.3);background:#fffef9;font:700 10.5px ui-monospace,SFMono-Regular,Consolas,monospace;color:#071321;transition:box-shadow .2s,border-color .2s;}'
    + '.ig-formula{margin:12px 0 0;font:700 10.5px ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.03em;color:#263541;}'
    + '.ig-formula b{color:#071321;font-weight:800;}'
    + '.ig-strip{display:flex;align-items:center;justify-content:center;gap:10px;flex-wrap:wrap;padding:12px 6px;margin-top:4px;border:1px solid #d8d3c4;background:#f7f5ef;}'
    + '.ig-mgroup{display:flex;flex-direction:column;align-items:center;gap:4px;min-width:0;}'
    + '.ig-cap{font:700 9px ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.08em;color:#65717c;text-transform:uppercase;text-align:center;white-space:nowrap;}'
    + '.ig-mat{display:flex;flex-direction:column;gap:2px;}'
    + '.ig-mrow{display:flex;gap:2px;align-items:center;}'
    + '.ig-rlab{flex:0 0 auto;text-align:right;padding-right:4px;font:700 8.5px ui-monospace,SFMono-Regular,Consolas,monospace;color:#8b949c;transition:color .2s;}'
    + '.ig-clab{flex:0 0 auto;text-align:center;font:700 8.5px ui-monospace,SFMono-Regular,Consolas,monospace;color:#8b949c;transition:color .2s;}'
    + '.ig-cell{box-sizing:border-box;min-width:30px;height:24px;display:grid;place-items:center;border:1px solid rgba(7,19,33,.3);background:#fffef9;font:700 10.5px ui-monospace,SFMono-Regular,Consolas,monospace;color:#071321;transition:opacity .2s,box-shadow .2s,border-color .2s;}'
    + '.ig-cell.ig-click{cursor:pointer;}'
    + '.ig-cell.ig-click:hover{border-color:#071321;}'
    + '.ig-op{font:800 15px ui-monospace,SFMono-Regular,Consolas,monospace;color:#071321;padding:0 1px;}'
    + '.ig-hl{border-color:#0b63f3!important;box-shadow:0 0 0 2px rgba(11,99,243,.38);}'
    + '.ig-rowhl{border-color:#0b63f3!important;box-shadow:0 0 0 2px rgba(11,99,243,.38);background:rgba(11,99,243,.13)!important;}'
    + '.ig-grn{border-color:#071321!important;box-shadow:0 0 0 3px rgba(200,255,71,.95);z-index:1;}'
    + '.ig-sel{background:#071321!important;color:#c8ff47!important;border-color:#071321!important;}'
    + '.ig-dim{opacity:.38;}'
    + '.ig-labh{color:#0b63f3!important;font-weight:800;}'
    + '.ig-readout{margin:10px 0 2px;padding:9px 12px;border:1px solid #d8d3c4;background:#f7f5ef;font:700 11px ui-monospace,SFMono-Regular,Consolas,monospace;color:#263541;line-height:1.9;}'
    + '.ig-readout b{color:#071321;font-weight:800;}'
    + '.ig-t{color:#45515c;white-space:nowrap;}'
    + '.ig-t.is-zero{color:#b9b2a4;}'
    + '.ig-div{margin:14px 0 0;padding-top:10px;border-top:1px dashed #d8d3c4;}'
    + '.ig-panel{display:none;}'
    + '.ig-panel.is-on{display:block;}'
    + '.ig-modes{display:flex;gap:6px;margin:2px 0 6px;}'
    + '.ig-modes .wg-button{flex:0 0 auto;}'
    + '.ig-modes .wg-button.is-on{background:#071321;color:#c8ff47;}';

  function mount(container) {
    var styleEl = document.createElement('style');
    styleEl.textContent = CSS;
    container.appendChild(styleEl);

    container.appendChild((function () { var d = document.createElement('div'); d.innerHTML =
      '<p class="wg-title">im2col：一次矩阵乘的正路与反路</p>' +
      '<p class="wg-sub">前向点 Y_col 看乘积来源；反向看 dW 与 col2im 梯度回传。</p>' +
      '<div class="ig-shape">X: (1,1,4,4) · W: (2,1,3,3) · Y: (1,2,2,2) · stride 1 无 padding · 展开 X_col: (4,9) · W_col: (9,2) · Y_col: (4,2)</div>' +
      '<div class="ig-modes">' +
        '<button type="button" class="wg-button is-on" data-role="vfwd">前向 · Y_col = X_col·W_col</button>' +
        '<button type="button" class="wg-button" data-role="vbwd">反向 · dW 与 col2im</button>' +
      '</div>' +
      '<div class="ig-top">' +
        '<div class="ig-gcol">' +
          '<div class="wg-label"><span>输入 X (4×4)</span><span data-role="xhint"></span></div>' +
          '<div class="ig-fixed" data-role="xhost"></div>' +
        '</div>' +
        '<div class="ig-gcol" data-role="dxcol" style="display:none;">' +
          '<div class="wg-label"><span>输入梯度 dX (4×4)</span><span>col2im 结果</span></div>' +
          '<div data-role="dxhost"></div>' +
        '</div>' +
        '<div class="ig-gcol">' +
          '<div class="wg-label"><span>核 W（2 个输出通道）</span><span data-role="khint"></span></div>' +
          '<div class="ig-kboxes" data-role="kboxes"></div>' +
        '</div>' +
      '</div>' +
      '<div class="ig-panel is-on" data-role="pfwd">' +
        '<p class="ig-formula">前向 · <b>Y_col = X_col · W_col</b> — (4,9)·(9,2) → (4,2)，内维 9 = 核的 tap 数</p>' +
        '<div class="ig-strip" data-role="fstrip"></div>' +
        '<div class="ig-readout" data-role="fread"></div>' +
        '<p class="wg-note">点击 Y_col 格子：高亮相乘的 X_col 行与 W_col 列；k = 行优先 tap 位。</p>' +
      '</div>' +
      '<div class="ig-panel" data-role="pbwd">' +
        '<p class="ig-formula">反向 · <b>dW = X_colᵀ · dY_col</b> — (9,4)·(4,2) → (9,2)，内维换成 4 = 窗口数</p>' +
        '<div class="ig-strip" data-role="bstrip"></div>' +
        '<div class="ig-readout" data-role="dread"></div>' +
        '<p class="wg-note">点击 dW 格子：来源为 X_colᵀ 第 k 行与 dY_col 第 o 列。</p>' +
        '<div class="ig-div">' +
          '<div class="wg-label"><span>col2im · 梯度叠回像素</span><span>点击 X 或 dX 的像素</span></div>' +
          '<div class="ig-readout" data-role="pread"></div>' +
          '<p class="wg-note">dX_col = W_col·dY_colᵀ；同一像素的格子累加成 dX。</p>' +
        '</div>' +
      '</div>' +
      '<p class="wg-note">全部数值为整数，可手算验证。</p>'; return d; })());

    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };

    var state = {
      view: 'fwd',            // 'fwd' | 'bwd'
      fwd: { r: 0, o: 0 },    // 选中的 Y_col 格
      dw: { k: 4, o: 1 },     // 选中的 dW 格
      px: { i: 1, j: 1 },     // col2im 选中的像素
      focus: 'px',            // 反向条带高亮归属：'dw' | 'px'
    };

    // ---- 输入 X 4×4 + 4 个复用的窗口框 ----
    var xhost = q('xhost');
    var xGrid = document.createElement('div');
    xGrid.className = 'ig-xgrid';
    xGrid.setAttribute('aria-label', '输入 4×4，点击选择窗口或像素');
    var xCells = [];
    for (var n = 0; n < 16; n++) {
      var xc = document.createElement('div');
      xc.className = 'ig-xcell ig-click';
      xGrid.appendChild(xc);
      xCells.push(xc);
    }
    xhost.appendChild(xGrid);
    var frames = [];
    for (var fn = 0; fn < 4; fn++) {
      var fr = document.createElement('div');
      fr.className = 'ig-frame';
      fr.style.display = 'none';
      var bd = document.createElement('span');
      bd.className = 'ig-fbadge';
      fr.appendChild(bd);
      xhost.appendChild(fr);
      frames.push(fr);
    }

    // ---- dX 4×4 ----
    var dxGrid = document.createElement('div');
    dxGrid.className = 'ig-xgrid';
    dxGrid.setAttribute('aria-label', '输入梯度 dX 4×4，点击查看累加路径');
    var dxCells = [];
    for (var dn = 0; dn < 16; dn++) {
      var dc = document.createElement('div');
      dc.className = 'ig-xcell ig-click';
      dxGrid.appendChild(dc);
      dxCells.push(dc);
    }
    q('dxhost').appendChild(dxGrid);

    // ---- 两个输出通道的 3×3 核 ----
    var kboxes = [], kCells = [[], []];
    for (var ch = 0; ch < 2; ch++) {
      (function (o2) {
        var box = document.createElement('div');
        box.className = 'ig-kbox';
        var cap = document.createElement('div');
        cap.className = 'ig-cap';
        cap.textContent = '核 o=' + o2;
        var g = document.createElement('div');
        g.className = 'ig-kgrid';
        for (var t = 0; t < 9; t++) {
          var kc = document.createElement('div');
          kc.className = 'ig-kcell';
          var v = WCOL[t][o2];
          kc.textContent = fmt(v);
          var col = divergent(v, 2);
          kc.style.background = col.bg;
          kc.style.color = col.fg;
          g.appendChild(kc);
          kCells[o2].push(kc);
        }
        box.appendChild(cap);
        box.appendChild(g);
        q('kboxes').appendChild(box);
        kboxes.push(box);
      }(ch));
    }

    // ---- 矩阵构建器 ----
    function buildGroup(host, cap) {
      var g = document.createElement('div');
      g.className = 'ig-mgroup';
      var cd = document.createElement('div');
      cd.className = 'ig-cap';
      cd.innerHTML = cap;
      g.appendChild(cd);
      host.appendChild(g);
      return g;
    }
    function addOp(host, sym) {
      var s = document.createElement('span');
      s.className = 'ig-op';
      s.textContent = sym;
      host.appendChild(s);
    }
    function buildMatrix(host, cfg) {
      var mat = document.createElement('div');
      mat.className = 'ig-mat';
      var colLabs = [], rowLabs = [], cells = [];
      var hrow = document.createElement('div');
      hrow.className = 'ig-mrow';
      var corner = document.createElement('span');
      corner.className = 'ig-rlab';
      corner.style.width = (cfg.rlabW || 34) + 'px';
      corner.innerHTML = cfg.corner || '';
      hrow.appendChild(corner);
      for (var hc = 0; hc < cfg.cols; hc++) {
        var cl = document.createElement('span');
        cl.className = 'ig-clab';
        cl.style.width = '30px';
        cl.textContent = cfg.clab ? cfg.clab(hc) : '';
        hrow.appendChild(cl);
        colLabs.push(cl);
      }
      mat.appendChild(hrow);
      for (var rr = 0; rr < cfg.rows; rr++) {
        var row = document.createElement('div');
        row.className = 'ig-mrow';
        var rl = document.createElement('span');
        rl.className = 'ig-rlab';
        rl.style.width = (cfg.rlabW || 34) + 'px';
        rl.textContent = cfg.rlab ? cfg.rlab(rr) : '';
        row.appendChild(rl);
        rowLabs.push(rl);
        cells.push([]);
        for (var cc = 0; cc < cfg.cols; cc++) {
          (function (a, b) {
            var cell = document.createElement('div');
            cell.className = 'ig-cell' + (cfg.click ? ' ig-click' : '');
            var v = cfg.val(a, b);
            cell.textContent = fmt(v);
            if (cfg.color) {
              var col = cfg.color(v);
              cell.style.background = col.bg;
              cell.style.color = col.fg;
            }
            if (cfg.click) {
              cell.addEventListener('click', function () { cfg.click(a, b); });
            }
            row.appendChild(cell);
            cells[a].push(cell);
          }(rr, cc));
        }
        mat.appendChild(row);
      }
      host.appendChild(mat);
      return { cells: cells, rowLabs: rowLabs, colLabs: colLabs };
    }
    function clearMat(m) {
      m.cells.forEach(function (rowc) {
        rowc.forEach(function (el) { el.classList.remove('ig-hl', 'ig-rowhl', 'ig-grn', 'ig-sel', 'ig-dim'); });
      });
      m.rowLabs.forEach(function (el) { el.classList.remove('ig-labh'); });
      m.colLabs.forEach(function (el) { el.classList.remove('ig-labh'); });
    }

    function winLab(n) { return '(' + WIN[n][0] + ',' + WIN[n][1] + ')'; }
    function tapLab(n) { return String(n); }
    function ocLab(n) { return 'o=' + n; }

    // ---- 前向条带：X_col × W_col = Y_col ----
    var fstrip = q('fstrip');
    var fX = buildMatrix(buildGroup(fstrip, 'X_col (4×9)'), {
      rows: 4, cols: 9, rlabW: 34,
      val: function (a, b) { return XCOL[a][b]; },
      rlab: winLab, clab: tapLab,
    });
    addOp(fstrip, '×');
    var fW = buildMatrix(buildGroup(fstrip, 'W_col (9×2)'), {
      rows: 9, cols: 2, rlabW: 18,
      val: function (a, b) { return WCOL[a][b]; },
      color: function (v) { return divergent(v, 2); },
      rlab: tapLab, clab: ocLab,
    });
    addOp(fstrip, '=');
    var fY = buildMatrix(buildGroup(fstrip, 'Y_col (4×2)'), {
      rows: 4, cols: 2, rlabW: 34,
      val: function (a, b) { return YCOL[a][b]; },
      color: function (v) { return divergent(v, 6); },
      rlab: winLab, clab: ocLab,
      click: function (a, b) { state.fwd = { r: a, o: b }; paint(); },
    });

    // ---- 反向条带：X_colᵀ × dY_col = dW ----
    var bstrip = q('bstrip');
    var bX = buildMatrix(buildGroup(bstrip, 'X_colᵀ (9×4)'), {
      rows: 9, cols: 4, rlabW: 18,
      val: function (a, b) { return XCOL[b][a]; },
      rlab: tapLab, clab: winLab,
    });
    addOp(bstrip, '×');
    var bD = buildMatrix(buildGroup(bstrip, 'dY_col (4×2)'), {
      rows: 4, cols: 2, rlabW: 34,
      val: function (a, b) { return DY[a][b]; },
      color: function (v) { return divergent(v, 2); },
      rlab: winLab, clab: ocLab,
    });
    addOp(bstrip, '=');
    var bW = buildMatrix(buildGroup(bstrip, 'dW (9×2)'), {
      rows: 9, cols: 2, rlabW: 18,
      val: function (a, b) { return DW[a][b]; },
      color: function (v) { return divergent(v, 7); },
      rlab: tapLab, clab: ocLab,
      click: function (a, b) { state.dw = { k: a, o: b }; state.focus = 'dw'; paint(); },
    });

    // ---- 定位窗口框 ----
    var pitch = null;
    function measure() {
      var a = xCells[0].getBoundingClientRect();
      var b = xCells[1].getBoundingClientRect();
      var d = xCells[4].getBoundingClientRect();
      pitch = { x: b.left - a.left, y: d.top - a.top, w: a.width, h: a.height };
    }
    function positionFrame(f, wr, wc) {
      if (!pitch) { measure(); }
      var host = xhost.getBoundingClientRect();
      var el = xCells[wr * 4 + wc].getBoundingClientRect();
      var gapX = pitch.x - pitch.w, gapY = pitch.y - pitch.h;
      f.style.left = (el.left - host.left - 2) + 'px';
      f.style.top = (el.top - host.top - 2) + 'px';
      f.style.width = (pitch.w * 3 + gapX * 2 + 4) + 'px';
      f.style.height = (pitch.h * 3 + gapY * 2 + 4) + 'px';
    }

    // ---- 各区域重绘 ----
    function paintFwd() {
      clearMat(fX); clearMat(fW); clearMat(fY);
      var rr = state.fwd.r, oo = state.fwd.o, a, b;
      for (a = 0; a < 4; a++) {
        for (b = 0; b < 9; b++) {
          if (a === rr) { fX.cells[a][b].classList.add('ig-rowhl'); }
          else { fX.cells[a][b].classList.add('ig-dim'); }
        }
      }
      for (a = 0; a < 9; a++) {
        for (b = 0; b < 2; b++) {
          if (b === oo) { fW.cells[a][b].classList.add('ig-hl'); }
          else { fW.cells[a][b].classList.add('ig-dim'); }
        }
      }
      for (a = 0; a < 4; a++) {
        for (b = 0; b < 2; b++) {
          if (a === rr && b === oo) { fY.cells[a][b].classList.add('ig-sel'); }
          else { fY.cells[a][b].classList.add('ig-dim'); }
        }
      }
      fX.rowLabs[rr].classList.add('ig-labh');
      fW.colLabs[oo].classList.add('ig-labh');
      fY.rowLabs[rr].classList.add('ig-labh');
      fY.colLabs[oo].classList.add('ig-labh');

      var terms = [];
      for (var t = 0; t < 9; t++) {
        var wv = WCOL[t][oo];
        var p = XCOL[rr][t] * wv;
        terms.push('<span class="ig-t' + (p === 0 ? ' is-zero' : '') + '">' + fmt(XCOL[rr][t]) + '·' + (wv < 0 ? '(' + fmt(wv) + ')' : fmt(wv)) + '</span>');
      }
      q('fread').innerHTML = 'Y_col[' + rr + ',' + oo + '] = 窗口 ' + winLab(rr) + ' 的 9 个数 · 通道 o=' + oo + ' 的核：<br>' +
        terms.join(' + ') + ' = <b>' + fmt(YCOL[rr][oo]) + '</b>';
    }

    function paintDwReadout() {
      var kk = state.dw.k, oo = state.dw.o;
      var terms = [];
      for (var t = 0; t < 4; t++) {
        var dv = DY[t][oo];
        var p = XCOL[t][kk] * dv;
        terms.push('<span class="ig-t' + (p === 0 ? ' is-zero' : '') + '">' + fmt(XCOL[t][kk]) + '·' + (dv < 0 ? '(' + fmt(dv) + ')' : fmt(dv)) + '</span>');
      }
      q('dread').innerHTML = 'dW[' + kk + ',' + oo + '] = tap (' + Math.floor(kk / 3) + ',' + (kk % 3) + ') · 通道 o=' + oo + '：4 个窗口在同一 tap 位置的输入 × 各自上游梯度<br>' +
        terms.join(' + ') + ' = <b>' + fmt(DW[kk][oo]) + '</b>' +
        (state.focus === 'px' ? ' <span class="ig-t">(点击 dW 任一格查看上图高亮来源)</span>' : '');
    }

    function paintBwdStrip() {
      clearMat(bX); clearMat(bD); clearMat(bW);
      var a, b;
      if (state.focus === 'dw') {
        var kk = state.dw.k, oo = state.dw.o;
        for (a = 0; a < 9; a++) {
          for (b = 0; b < 4; b++) {
            if (a === kk) { bX.cells[a][b].classList.add('ig-rowhl'); }
            else { bX.cells[a][b].classList.add('ig-dim'); }
          }
        }
        for (a = 0; a < 4; a++) {
          for (b = 0; b < 2; b++) {
            if (b === oo) { bD.cells[a][b].classList.add('ig-hl'); }
            else { bD.cells[a][b].classList.add('ig-dim'); }
          }
        }
        for (a = 0; a < 9; a++) {
          for (b = 0; b < 2; b++) {
            if (a === kk && b === oo) { bW.cells[a][b].classList.add('ig-sel'); }
            else { bW.cells[a][b].classList.add('ig-dim'); }
          }
        }
        bX.rowLabs[kk].classList.add('ig-labh');
        bD.colLabs[oo].classList.add('ig-labh');
      } else {
        var wins = windowsOf(state.px.i, state.px.j);
        var colSet = {};
        wins.forEach(function (w) { colSet[w.r] = w; });
        for (a = 0; a < 9; a++) {
          for (b = 0; b < 4; b++) {
            if (colSet[b] !== undefined) { bX.cells[a][b].classList.add('ig-rowhl'); }
            else { bX.cells[a][b].classList.add('ig-dim'); }
          }
        }
        wins.forEach(function (w) {
          bX.cells[w.k][w.r].classList.remove('ig-rowhl');
          bX.cells[w.k][w.r].classList.add('ig-grn');
          bX.colLabs[w.r].classList.add('ig-labh');
        });
        for (a = 0; a < 4; a++) {
          for (b = 0; b < 2; b++) {
            if (colSet[a] !== undefined) { bD.cells[a][b].classList.add('ig-hl'); }
            else { bD.cells[a][b].classList.add('ig-dim'); }
          }
        }
      }
    }

    function paintPxReadout() {
      var pi = state.px.i, pj = state.px.j;
      var wins = windowsOf(pi, pj);
      q('pread').innerHTML = '像素 (' + pi + ',' + pj + ') 被 ' + wins.length + ' 个窗口覆盖：' +
        wins.map(function (w) { return 'X_col 行' + w.r + ' 的 k=' + w.k; }).join(' · ') +
        '（同一像素在不同窗口的 tap 位不同）<br>' +
        wins.map(function (w) { return 'dX_col[' + w.r + ',' + w.k + ']'; }).join(' + ') + ' = ' +
        wins.map(function (w) { return fmt(DXCOL[w.r][w.k]); }).join(' + ') + ' = <b>' + fmt(DX[pi][pj]) + '</b>';
    }

    function paintX() {
      var wins = windowsOf(state.px.i, state.px.j);
      for (var n2 = 0; n2 < 16; n2++) {
        var el = xCells[n2];
        var ii = Math.floor(n2 / 4), jj = n2 % 4;
        el.textContent = String(X[ii][jj]);
        el.style.background = inputColor(X[ii][jj]);
        el.style.color = X[ii][jj] >= 3 ? CARD : INK;
        el.classList.remove('ig-grn', 'ig-dim');
        if (state.view === 'bwd') {
          var inWin = wins.some(function (w) {
            var wr = WIN[w.r][0], wc = WIN[w.r][1];
            return ii >= wr && ii <= wr + 2 && jj >= wc && jj <= wc + 2;
          });
          if (!inWin) { el.classList.add('ig-dim'); }
          if (ii === state.px.i && jj === state.px.j) { el.classList.add('ig-grn'); }
        }
      }
      if (state.view === 'fwd') {
        frames.forEach(function (f, n3) { f.style.display = n3 === 0 ? '' : 'none'; });
        frames[0].querySelector('.ig-fbadge').style.display = 'none';
        positionFrame(frames[0], WIN[state.fwd.r][0], WIN[state.fwd.r][1]);
        q('xhint').textContent = '蓝框 = 窗口 ' + winLab(state.fwd.r) + ' · 点击换窗口';
      } else {
        for (var m = 0; m < 4; m++) {
          var f2 = frames[m];
          if (m < wins.length) {
            f2.style.display = '';
            var bd2 = f2.querySelector('.ig-fbadge');
            bd2.style.display = '';
            bd2.textContent = '行' + wins[m].r;
            positionFrame(f2, WIN[wins[m].r][0], WIN[wins[m].r][1]);
          } else { f2.style.display = 'none'; }
        }
        q('xhint').textContent = '绿格 = 像素 (' + state.px.i + ',' + state.px.j + ') · 蓝框 = 覆盖它的窗口';
      }
    }

    function paintDx() {
      for (var n2 = 0; n2 < 16; n2++) {
        var el = dxCells[n2];
        var ii = Math.floor(n2 / 4), jj = n2 % 4;
        var v = DX[ii][jj];
        el.textContent = fmt(v);
        var col = divergent(v, 7);
        el.style.background = col.bg;
        el.style.color = col.fg;
        el.classList.remove('ig-sel');
        if (state.view === 'bwd' && ii === state.px.i && jj === state.px.j) { el.classList.add('ig-sel'); }
      }
    }

    function paintKernels() {
      for (var o2 = 0; o2 < 2; o2++) {
        var on = state.view === 'fwd' ? (state.fwd.o === o2) : (state.focus === 'dw' && state.dw.o === o2);
        kboxes[o2].classList.toggle('is-on', on);
        for (var t = 0; t < 9; t++) {
          kCells[o2][t].classList.remove('ig-grn');
          if (state.view === 'bwd' && state.focus === 'dw' && state.dw.k === t && state.dw.o === o2) {
            kCells[o2][t].classList.add('ig-grn');
          }
        }
      }
      q('khint').textContent = state.view === 'fwd' ? '蓝框 = 参与当前点积的通道' : '绿格 = 当前 dW 的 tap 位';
    }

    function paint() {
      q('vfwd').classList.toggle('is-on', state.view === 'fwd');
      q('vbwd').classList.toggle('is-on', state.view === 'bwd');
      q('pfwd').classList.toggle('is-on', state.view === 'fwd');
      q('pbwd').classList.toggle('is-on', state.view === 'bwd');
      q('dxcol').style.display = state.view === 'bwd' ? '' : 'none';
      if (state.view === 'fwd') { paintFwd(); } else { paintBwdStrip(); paintDwReadout(); paintPxReadout(); }
      paintX();
      paintDx();
      paintKernels();
    }

    function pickFromCell(ii, jj) {
      if (state.view === 'fwd') {
        var wr = Math.min(ii, 1), wc = Math.min(jj, 1);
        for (var m2 = 0; m2 < 4; m2++) {
          if (WIN[m2][0] === wr && WIN[m2][1] === wc) { state.fwd.r = m2; }
        }
      } else {
        state.px = { i: ii, j: jj };
        state.focus = 'px';
      }
      paint();
    }
    xGrid.addEventListener('click', function (e) {
      var n2 = xCells.indexOf(e.target);
      if (n2 < 0) { return; }
      pickFromCell(Math.floor(n2 / 4), n2 % 4);
    });
    dxGrid.addEventListener('click', function (e) {
      var n2 = dxCells.indexOf(e.target);
      if (n2 < 0) { return; }
      if (state.view === 'bwd') {
        state.px = { i: Math.floor(n2 / 4), j: n2 % 4 };
        state.focus = 'px';
        paint();
      }
    });

    q('vfwd').addEventListener('click', function () { state.view = 'fwd'; paint(); });
    q('vbwd').addEventListener('click', function () { state.view = 'bwd'; paint(); });

    var onResize = function () { pitch = null; paintX(); };
    window.addEventListener('resize', onResize);

    paint();

    return function dispose() {
      window.removeEventListener('resize', onResize);
      if (styleEl.parentNode) { styleEl.parentNode.removeChild(styleEl); }
    };
  }

  window.DLWidget.register('im2col-grad', mount);
})();
