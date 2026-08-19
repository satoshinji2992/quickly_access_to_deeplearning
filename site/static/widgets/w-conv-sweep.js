/* conv-sweep — 拖动 3×3 卷积核扫过 8×8 输入图：逐格补满 6×6 特征图，并联动 im2col 展开。 */
(function () {
  'use strict';

  var INK = '#071321', BLUE = '#0b63f3', RED = '#c0392b', PAPER = '#f7f5ef', CARD = '#fffef9';

  // 8×8 输入：左侧暗、第 4-5 列是竖直亮条、右侧回落，再叠少量噪声。
  var INPUT = [
    [2, 0, 1, 4, 8, 9, 2, 1],
    [0, 2, 1, 5, 9, 8, 1, 2],
    [2, 1, 0, 4, 8, 9, 2, 0],
    [1, 2, 2, 5, 9, 8, 1, 1],
    [0, 1, 2, 4, 8, 9, 1, 2],
    [2, 0, 1, 5, 9, 8, 2, 1],
    [1, 2, 0, 4, 8, 9, 1, 0],
    [0, 1, 2, 5, 9, 8, 2, 2],
  ];

  var PRESETS = [
    { name: '竖直边缘', k: [[1, 0, -1], [1, 0, -1], [1, 0, -1]] },
    { name: '水平边缘', k: [[1, 1, 1], [0, 0, 0], [-1, -1, -1]] },
    { name: '锐化', k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
    { name: '平滑', k: [[1, 1, 1], [1, 1, 1], [1, 1, 1]] },
  ];

  var CSS = ''
    + '.cs-shape{margin:2px 0 10px;font:700 10px ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.06em;color:#45515c;}'
    + '.cs-main{display:flex;gap:14px;align-items:flex-start;}'
    + '.cs-inwrap{flex:1 1 auto;min-width:0;max-width:262px;}'
    + '.cs-fixed{position:relative;}'
    + '.cs-frame{position:absolute;box-sizing:border-box;border:2px solid #0b63f3;box-shadow:0 0 0 3px rgba(11,99,243,.18);pointer-events:none;z-index:2;transition:left .18s ease,top .18s ease,width .18s ease,height .18s ease;}'
    + '.cs-kcol{flex:0 0 112px;display:flex;flex-direction:column;}'
    + '.cs-presets{display:flex;flex-direction:column;gap:4px;margin-top:10px;}'
    + '.cs-preset{width:100%;padding:6px 8px;border:1px solid #071321;background:#fffef9;font:700 10px ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.04em;color:#071321;cursor:pointer;}'
    + '.cs-preset:hover{background:#071321;color:#fffef9;}'
    + '.cs-preset.is-on{background:#071321;color:#c8ff47;}'
    + '.cs-knote{margin:8px 0 0;font-size:10.5px;line-height:1.5;color:#65717c;}'
    + '.cs-outwrap{width:100%;max-width:248px;}'
    + '.cs-grid8{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));touch-action:none;}'
    + '.cs-grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:4px;}'
    + '.cs-grid6{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));}'
    + '.cs-out .wg-cell{transition:box-shadow .45s ease,border-color .45s ease;}'
    + '.cs-fresh{border-color:#0b63f3!important;box-shadow:0 0 0 3px rgba(11,99,243,.42);}'
    + '.cs-im{display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:10px;padding:12px 8px;border:1px solid #d8d3c4;background:#f7f5ef;}'
    + '.cs-im-group{display:flex;flex-direction:column;align-items:center;gap:4px;}'
    + '.cs-im-cap{font:700 9px ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.08em;color:#65717c;text-transform:uppercase;}'
    + '.cs-im-row{display:flex;gap:2px;}'
    + '.cs-im-col{display:flex;flex-direction:column;gap:2px;}'
    + '.cs-im-cell{min-width:19px;height:17px;padding:0 2px;display:grid;place-items:center;box-sizing:border-box;border:1px solid rgba(7,19,33,.28);background:#fffef9;font:700 10px ui-monospace,SFMono-Regular,Consolas,monospace;color:#071321;}'
    + '.cs-im-op{font:800 16px ui-monospace,SFMono-Regular,Consolas,monospace;color:#071321;}'
    + '.cs-im-res b{display:block;font:800 18px ui-monospace,SFMono-Regular,Consolas,monospace;color:#071321;}';

  function dotAt(r, c, k) {
    var s = 0;
    for (var i = 0; i < 3; i++) {
      for (var j = 0; j < 3; j++) { s += INPUT[r + i][c + j] * k[i][j]; }
    }
    return s;
  }

  function copyKernel(k) {
    return k.map(function (row) { return row.slice(); });
  }

  function inputColor(v) {
    var a = v === 0 ? 0.05 : 0.12 + 0.78 * (v / 9);
    return 'rgba(11,99,243,' + a.toFixed(3) + ')';
  }

  // 发散配色：正值偏红、负值偏蓝、零为纸色，透明度随绝对值。
  function divergent(v, scale) {
    var t = Math.max(0, Math.min(1, Math.abs(v) / (scale || 1)));
    if (v > 0) { return { bg: 'rgba(192,57,43,' + (0.10 + 0.72 * t).toFixed(3) + ')', fg: t > 0.55 ? '#fffef9' : INK }; }
    if (v < 0) { return { bg: 'rgba(11,99,243,' + (0.10 + 0.72 * t).toFixed(3) + ')', fg: t > 0.55 ? '#fffef9' : INK }; }
    return { bg: PAPER, fg: '#65717c' };
  }

  function mount(container) {
    var styleEl = document.createElement('style');
    styleEl.textContent = CSS;
    container.appendChild(styleEl);

    container.appendChild((function () { var d = document.createElement('div'); d.innerHTML =
      '<p class="wg-title">卷积扫过图像</p>' +
      '<p class="wg-sub">点击或拖动 3×3 窗口，每停一处补一格输出。</p>' +
      '<div class="cs-shape">X: (1,1,8,8) · W: (1,1,3,3) · Y: (1,1,6,6)</div>' +
      '<div class="cs-main">' +
        '<div class="cs-inwrap">' +
          '<div class="wg-label"><span>输入 X</span><span data-role="pos"></span></div>' +
          '<div class="cs-fixed" data-role="inhost"></div>' +
        '</div>' +
        '<div class="cs-kcol">' +
          '<div class="wg-label"><span>卷积核 W</span></div>' +
          '<div data-role="khost"></div>' +
          '<div class="cs-presets" data-role="presets"></div>' +
          '<p class="cs-knote">点击 +1，Shift 点击 −1（范围 −3…4）</p>' +
        '</div>' +
      '</div>' +
      '<div class="wg-block">' +
        '<div class="wg-label"><span>输出特征图 Y</span><span data-role="count"></span></div>' +
        '<div class="cs-out cs-outwrap cs-grid6" data-role="outhost"></div>' +
      '</div>' +
      '<div class="wg-controls">' +
        '<button type="button" class="wg-button is-primary" data-role="auto">自动扫描</button>' +
        '<button type="button" class="wg-button" data-role="all">显示全部</button>' +
        '<button type="button" class="wg-button" data-role="reset">重置</button>' +
      '</div>' +
      '<p class="wg-note">自动扫描每 300ms 前进一格；改核会清空输出。</p>' +
      '<div class="wg-block">' +
        '<div class="wg-label"><span>当前窗口展开 · im2col</span><span>Y[r,c] = X_col · W_col</span></div>' +
        '<div class="cs-im">' +
          '<div class="cs-im-group"><div class="cs-im-cap">X 窗口 (1×9)</div><div class="cs-im-row" data-role="imrow"></div></div>' +
          '<span class="cs-im-op">·</span>' +
          '<div class="cs-im-group"><div class="cs-im-cap">W 核 (9×1)</div><div class="cs-im-col" data-role="imcol"></div></div>' +
          '<span class="cs-im-op">=</span>' +
          '<div class="cs-im-group"><div class="cs-im-cap">点积</div><b data-role="imval">0</b></div>' +
        '</div>' +
      '</div>'; return d; })());

    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };

    var state = {
      r: 0, c: 0, scan: 0,
      kernel: copyKernel(PRESETS[0].k),
      preset: 0,
      computed: [],
      timer: null,
    };
    var pitch = null;
    var scale = 1;
    var freshCell = null;
    var dragging = false;

    // ---- 输入 8×8 ----
    var inputGrid = document.createElement('div');
    inputGrid.className = 'wg-grid cs-grid8';
    inputGrid.setAttribute('aria-label', '输入图像 8×8，点击或拖动选择卷积窗口');
    var inputCells = [];
    for (var i = 0; i < 8; i++) {
      for (var j = 0; j < 8; j++) {
        var cell = document.createElement('div');
        cell.className = 'wg-cell';
        var v = INPUT[i][j];
        cell.textContent = String(v);
        cell.style.background = inputColor(v);
        cell.style.color = v >= 6 ? '#fffef9' : INK;
        inputGrid.appendChild(cell);
        inputCells.push(cell);
      }
    }
    var frame = document.createElement('div');
    frame.className = 'cs-frame';
    var inhost = q('inhost');
    inhost.appendChild(inputGrid);
    inhost.appendChild(frame);

    // ---- 卷积核 3×3（可编辑） ----
    var kgrid = document.createElement('div');
    kgrid.className = 'wg-grid cs-grid3';
    kgrid.setAttribute('aria-label', '3×3 卷积核，点击加一');
    var kCells = [];
    for (var ki = 0; ki < 3; ki++) {
      for (var kj = 0; kj < 3; kj++) {
        (function (ri, cj) {
          var kc = document.createElement('div');
          kc.className = 'wg-cell';
          kc.title = '点击 +1 · Shift 点击 −1';
          kc.addEventListener('click', function (e) {
            var next = state.kernel[ri][cj] + (e.shiftKey ? -1 : 1);
            next = Math.max(-3, Math.min(4, next));
            if (next === state.kernel[ri][cj]) { return; }
            state.kernel[ri][cj] = next;
            state.preset = -1;
            paintPresets();
            onKernelChange();
          });
          kgrid.appendChild(kc);
          kCells.push(kc);
        }(ki, kj));
      }
    }
    q('khost').appendChild(kgrid);

    // ---- 输出 6×6 ----
    var outGrid = q('outhost');
    var outCells = [];
    for (var oi = 0; oi < 36; oi++) {
      var oc = document.createElement('div');
      oc.className = 'wg-cell';
      outGrid.appendChild(oc);
      outCells.push(oc);
    }

    // ---- im2col 展开 ----
    var imRow = q('imrow'), imCol = q('imcol'), imVal = q('imval');
    var imRowCells = [], imColCells = [];
    for (var m = 0; m < 9; m++) {
      var rc = document.createElement('div'); rc.className = 'cs-im-cell'; imRow.appendChild(rc); imRowCells.push(rc);
      var cc = document.createElement('div'); cc.className = 'cs-im-cell'; imCol.appendChild(cc); imColCells.push(cc);
    }

    // ---- 预设按钮 ----
    var presetBtns = [];
    PRESETS.forEach(function (p, idx) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'cs-preset';
      b.textContent = p.name;
      b.addEventListener('click', function () {
        state.kernel = copyKernel(p.k);
        state.preset = idx;
        paintPresets();
        onKernelChange();
      });
      q('presets').appendChild(b);
      presetBtns.push(b);
    });

    function paintPresets() {
      presetBtns.forEach(function (b, idx) { b.classList.toggle('is-on', idx === state.preset); });
    }

    function computeScale() {
      var mx = 1;
      for (var r = 0; r < 6; r++) {
        for (var c = 0; c < 6; c++) { mx = Math.max(mx, Math.abs(dotAt(r, c, state.kernel))); }
      }
      return mx;
    }

    function paintKernel() {
      var mx = 1;
      state.kernel.forEach(function (row) { row.forEach(function (v) { mx = Math.max(mx, Math.abs(v)); }); });
      for (var n = 0; n < 9; n++) {
        var v = state.kernel[Math.floor(n / 3)][n % 3];
        var col = divergent(v, mx);
        kCells[n].textContent = String(v);
        kCells[n].style.background = col.bg;
        kCells[n].style.color = col.fg;
      }
    }

    function clearOutputs() {
      state.computed = [];
      outCells.forEach(function (oc) {
        oc.textContent = '';
        oc.style.background = CARD;
        oc.style.color = INK;
        oc.classList.remove('cs-fresh');
      });
      freshCell = null;
      scale = computeScale();
      updateCount();
    }

    function updateCount() {
      var n = 0;
      state.computed.forEach(function (b) { if (b) { n += 1; } });
      q('count').textContent = '已算 ' + n + '/36';
    }

    function markFresh(cellEl) {
      if (freshCell && freshCell !== cellEl) { freshCell.classList.remove('cs-fresh'); }
      freshCell = cellEl;
      cellEl.classList.remove('cs-fresh');
      void cellEl.offsetWidth;  // 重启过渡，重复填充同一格也有动画
      cellEl.classList.add('cs-fresh');
    }

    function fillOutput(r, c, fresh) {
      var v = dotAt(r, c, state.kernel);
      var cellEl = outCells[r * 6 + c];
      cellEl.textContent = String(v);
      var col = divergent(v, scale);
      cellEl.style.background = col.bg;
      cellEl.style.color = col.fg;
      state.computed[r * 6 + c] = true;
      if (fresh) { markFresh(cellEl); }
      updateCount();
    }

    function updateIm2col() {
      var flatIn = [], flatK = [];
      for (var a = 0; a < 3; a++) {
        for (var b = 0; b < 3; b++) {
          flatIn.push(INPUT[state.r + a][state.c + b]);
          flatK.push(state.kernel[a][b]);
        }
      }
      var kmax = 1;
      flatK.forEach(function (v) { kmax = Math.max(kmax, Math.abs(v)); });
      for (var n = 0; n < 9; n++) {
        imRowCells[n].textContent = String(flatIn[n]);
        imRowCells[n].style.background = inputColor(flatIn[n]);
        imRowCells[n].style.color = flatIn[n] >= 6 ? '#fffef9' : INK;
        var col = divergent(flatK[n], kmax);
        imColCells[n].textContent = String(flatK[n]);
        imColCells[n].style.background = col.bg;
        imColCells[n].style.color = col.fg;
      }
      var v = dotAt(state.r, state.c, state.kernel);
      imVal.textContent = String(v);
      imVal.style.color = v > 0 ? RED : (v < 0 ? BLUE : '#65717c');
    }

    function measure() {
      var a = inputCells[0].getBoundingClientRect();
      var b = inputCells[1].getBoundingClientRect();
      var d = inputCells[8].getBoundingClientRect();
      pitch = { x: b.left - a.left, y: d.top - a.top, w: a.width, h: a.height };
    }

    function positionFrame() {
      if (!pitch) { measure(); }
      var host = inhost.getBoundingClientRect();
      var cellEl = inputCells[state.r * 8 + state.c].getBoundingClientRect();
      var gapX = pitch.x - pitch.w, gapY = pitch.y - pitch.h;
      frame.style.left = (cellEl.left - host.left - 2) + 'px';
      frame.style.top = (cellEl.top - host.top - 2) + 'px';
      frame.style.width = (pitch.w * 3 + gapX * 2 + 4) + 'px';
      frame.style.height = (pitch.h * 3 + gapY * 2 + 4) + 'px';
    }

    function setCursor(r, c, fill) {
      state.r = Math.max(0, Math.min(5, r));
      state.c = Math.max(0, Math.min(5, c));
      state.scan = state.r * 6 + state.c;
      q('pos').textContent = '窗口 (' + state.r + ',' + state.c + ')';
      positionFrame();
      updateIm2col();
      if (fill) { fillOutput(state.r, state.c, true); }
    }

    function onKernelChange() {
      clearOutputs();
      paintKernel();
      updateIm2col();
    }

    function cellFromEvent(e) {
      if (!pitch) { measure(); }
      var rect = inputGrid.getBoundingClientRect();
      var col = Math.floor((e.clientX - rect.left + 1.5) / pitch.x);
      var row = Math.floor((e.clientY - rect.top + 1.5) / pitch.y);
      return { row: Math.min(Math.max(0, Math.min(7, row)), 5), col: Math.min(Math.max(0, Math.min(7, col)), 5) };
    }

    inputGrid.addEventListener('pointerdown', function (e) {
      e.preventDefault();
      dragging = true;
      try { inputGrid.setPointerCapture(e.pointerId); } catch (err) { /* 旧浏览器忽略 */ }
      var p = cellFromEvent(e);
      setCursor(p.row, p.col, true);
    });
    inputGrid.addEventListener('pointermove', function (e) {
      if (!dragging) { return; }
      var p = cellFromEvent(e);
      if (p.row !== state.r || p.col !== state.c) { setCursor(p.row, p.col, true); }
    });
    function endDrag() { dragging = false; }
    inputGrid.addEventListener('pointerup', endDrag);
    inputGrid.addEventListener('pointercancel', endDrag);

    function stopAuto() {
      if (state.timer) { window.clearInterval(state.timer); state.timer = null; }
      q('auto').textContent = '自动扫描';
    }

    q('auto').addEventListener('click', function () {
      if (state.timer) { stopAuto(); return; }
      if (state.scan >= 35) { state.scan = -1; }
      this.textContent = '暂停';
      state.timer = window.setInterval(function () {
        if (state.scan >= 35) { stopAuto(); return; }
        var n = state.scan + 1;
        setCursor(Math.floor(n / 6), n % 6, true);
      }, 300);
    });

    q('all').addEventListener('click', function () {
      stopAuto();
      for (var r = 0; r < 6; r++) {
        for (var c = 0; c < 6; c++) { fillOutput(r, c, false); }
      }
    });

    q('reset').addEventListener('click', function () {
      stopAuto();
      state.kernel = copyKernel(PRESETS[0].k);
      state.preset = 0;
      paintPresets();
      clearOutputs();
      paintKernel();
      setCursor(0, 0, false);
    });

    var onResize = function () { pitch = null; positionFrame(); };
    window.addEventListener('resize', onResize);

    // 初始：预填 (0,0) 与 (3,3) 两个示例，窗口停在 (3,3)。
    paintPresets();
    clearOutputs();
    paintKernel();
    setCursor(0, 0, true);
    setCursor(3, 3, true);

    return function dispose() {
      stopAuto();
      window.removeEventListener('resize', onResize);
      if (styleEl.parentNode) { styleEl.parentNode.removeChild(styleEl); }
    };
  }

  window.DLWidget.register('conv-sweep', mount);
})();
