/* pool-sweep — 2×2 池化窗口扫过 8×8 输入：Max/Avg 前向取数，反向视图展示 dY 的梯度路由。 */
(function () {
  'use strict';

  var INK = '#071321', BLUE = '#0b63f3', RED = '#c0392b', PAPER = '#f7f5ef', CARD = '#fffef9';

  // 8×8 输入：中部有亮斑（8/9），W(1,2) 窗口内两处 9 并列（演示行优先 tie-break）。
  var INPUT = [
    [1, 3, 2, 0, 5, 1, 0, 2],
    [2, 1, 4, 3, 1, 6, 2, 0],
    [0, 2, 1, 5, 8, 9, 1, 2],
    [3, 1, 2, 4, 9, 7, 0, 1],
    [1, 4, 6, 2, 3, 0, 5, 2],
    [2, 2, 3, 1, 4, 5, 1, 3],
    [0, 1, 2, 5, 1, 3, 6, 2],
    [3, 0, 4, 2, 2, 1, 5, 4],
  ];

  var CSS = ''
    + '.ps-shape{margin:2px 0 10px;font:700 10px ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.06em;color:#45515c;}'
    + '.ps-main{display:flex;gap:16px;align-items:flex-start;}'
    + '.ps-inwrap{flex:1 1 auto;min-width:0;max-width:272px;}'
    + '.ps-fixed{position:relative;}'
    + '.ps-frame{position:absolute;box-sizing:border-box;border:2px solid #0b63f3;box-shadow:0 0 0 3px rgba(11,99,243,.18);pointer-events:none;z-index:2;transition:left .18s ease,top .18s ease;}'
    + '.ps-side{flex:0 0 236px;display:flex;flex-direction:column;min-width:0;}'
    + '.ps-modes{display:flex;gap:6px;margin:2px 0 6px;}'
    + '.ps-modes .wg-button{flex:1 1 auto;}'
    + '.ps-modes .wg-button.is-on{background:#071321;color:#c8ff47;}'
    + '.ps-grid8{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));touch-action:none;}'
    + '.ps-grid4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:4px;max-width:212px;}'
    + '.ps-out .wg-cell{transition:box-shadow .45s ease,border-color .45s ease;cursor:pointer;}'
    + '.ps-fresh{border-color:#0b63f3!important;box-shadow:0 0 0 3px rgba(11,99,243,.42);}'
    + '.ps-rel{position:relative;}'
    + '.ps-badge{position:absolute;z-index:1;background:rgba(255,254,249,.92);padding:0 1px;font:800 7.5px ui-monospace,SFMono-Regular,Consolas,monospace;line-height:1.2;pointer-events:none;}'
    + '.ps-bamt{right:1px;bottom:1px;color:#c0392b;}'
    + '.ps-bq{left:1px;top:1px;color:#0b63f3;}'
    + '.ps-win-max{box-shadow:0 0 0 3px rgba(200,255,71,.95);z-index:1;}'
    + '.ps-win-avg{outline:2px dashed rgba(255,254,249,.95);outline-offset:-3px;}'
    + '.ps-dim{opacity:.45;}'
    + '.ps-legend{margin-top:7px;font:700 9px ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.06em;color:#65717c;text-transform:uppercase;}'
    + '.ps-sidenote{margin:8px 0 0;font-size:10.5px;line-height:1.5;color:#65717c;}'
    + '.ps-readout{margin:12px 0 2px;padding:9px 12px;border:1px solid #d8d3c4;background:#f7f5ef;font:700 11.5px ui-monospace,SFMono-Regular,Consolas,monospace;color:#263541;line-height:1.7;}'
    + '.ps-readout b{color:#071321;font-weight:800;}'
    + '.ps-warn{color:#c0392b;}'
    + '.ps-back{margin:12px 0 0;padding-top:10px;border-top:1px dashed #d8d3c4;}'
    + '@media (max-width:700px){.ps-main{flex-direction:column;}.ps-side{flex:1 1 auto;}}';

  // 窗口 (wr,wc) 覆盖的 4 个输入格（行优先）。
  function windowCells(wr, wc) {
    var out = [];
    for (var i = 0; i < 2; i++) {
      for (var j = 0; j < 2; j++) {
        var r = wr * 2 + i, c = wc * 2 + j;
        out.push({ r: r, c: c, v: INPUT[r][c] });
      }
    }
    return out;
  }

  // 行优先第一个最大值（与 np.argmax 的并列规则一致）。
  function argmaxCell(cells) {
    var best = cells[0];
    for (var n = 1; n < cells.length; n++) {
      if (cells[n].v > best.v) { best = cells[n]; }
    }
    return best;
  }

  function maxInfo(cells) {
    var m = argmaxCell(cells).v, ties = 0;
    for (var n = 0; n < cells.length; n++) { if (cells[n].v === m) { ties += 1; } }
    return { max: m, ties: ties };
  }

  function avgOf(cells) {
    var s = 0;
    for (var n = 0; n < cells.length; n++) { s += cells[n].v; }
    return s / cells.length;
  }

  function fmt(x) {
    var r = Math.round(x * 100) / 100;
    if (Number.isInteger(r)) { return String(r); }
    return r.toFixed(2).replace(/0+$/, '').replace(/\.$/, '');
  }

  function inputColor(v) {
    var a = v === 0 ? 0.05 : 0.12 + 0.78 * (v / 9);
    return 'rgba(11,99,243,' + a.toFixed(3) + ')';
  }

  function mount(container) {
    var styleEl = document.createElement('style');
    styleEl.textContent = CSS;
    container.appendChild(styleEl);

    container.appendChild((function () { var d = document.createElement('div'); d.innerHTML =
      '<p class="wg-title">池化：窗口取数与梯度路由</p>' +
      '<p class="wg-sub">点击或拖动移动 2×2 窗口；「反向视图」看 dY → dX。</p>' +
      '<div class="ps-shape">X: (1,1,8,8) → Y: (1,1,4,4) · 窗口 2×2 · stride 2 · 无参数</div>' +
      '<div class="ps-main">' +
        '<div class="ps-inwrap">' +
          '<div class="wg-label"><span>输入 X</span><span data-role="pos"></span></div>' +
          '<div class="ps-fixed" data-role="inhost"></div>' +
          '<div class="ps-legend" data-role="legend"></div>' +
        '</div>' +
        '<div class="ps-side">' +
          '<div class="wg-label"><span>模式</span><span>取最大 / 取平均</span></div>' +
          '<div class="ps-modes">' +
            '<button type="button" class="wg-button" data-role="mode-max">MaxPool</button>' +
            '<button type="button" class="wg-button" data-role="mode-avg">AvgPool</button>' +
          '</div>' +
          '<div class="wg-label"><span data-role="outlabel">输出 Y (1,1,4,4)</span><span data-role="count"></span></div>' +
          '<div class="ps-out ps-grid4" data-role="outhost" aria-label="4×4 输出网格"></div>' +
          '<p class="ps-sidenote" data-role="sidenote"></p>' +
        '</div>' +
      '</div>' +
      '<div class="ps-readout" data-role="fwdread"></div>' +
      '<div class="wg-controls">' +
        '<button type="button" class="wg-button is-primary" data-role="auto">自动扫描</button>' +
        '<button type="button" class="wg-button" data-role="all">显示全部</button>' +
        '<button type="button" class="wg-button" data-role="reset">重置</button>' +
        '<button type="button" class="wg-button" data-role="view">反向视图</button>' +
      '</div>' +
      '<div class="ps-back" data-role="backpanel" style="display:none;">' +
        '<div class="wg-label"><span>反向：dY → dX</span><span>dY: (1,1,4,4) → dX: (1,1,8,8)</span></div>' +
        '<div class="ps-readout" data-role="backread"></div>' +
        '<p class="wg-note">Max：梯度只进胜出格；Avg：四格各 ¼。</p>' +
      '</div>' +
      '<p class="wg-note">并列最大值取行优先第一个，同 np.argmax 规则。</p>'; return d; })());

    // 反向路由表：输入格下标 → 收到的 dX 量。
    function routesFor(wr, wc, mode) {
      var cells = windowCells(wr, wc), map = {};
      if (mode === 'max') {
        var a = argmaxCell(cells);
        map[a.r * 8 + a.c] = 1;
      } else {
        for (var n = 0; n < cells.length; n++) { map[cells[n].r * 8 + cells[n].c] = 0.25; }
      }
      return map;
    }

    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };

    var state = {
      wr: 1, wc: 2,          // 当前窗口 = 反向选中的 dY 格
      mode: 'max',           // 'max' | 'avg'
      view: 'fwd',           // 'fwd' | 'bwd'
      computed: [],          // 16 个输出格是否已算
      backAll: false,        // 反向视图：dY 是否全 1
      scan: 6,
      timer: null,
    };

    var pitch = null;
    var freshCell = null;
    var dragging = false;

    // ---- 输入 8×8 ----
    var inputGrid = document.createElement('div');
    inputGrid.className = 'wg-grid ps-grid8';
    inputGrid.setAttribute('aria-label', '输入 8×8，点击或拖动选择池化窗口');
    var inputCells = [];
    for (var i = 0; i < 8; i++) {
      for (var j = 0; j < 8; j++) {
        var cell = document.createElement('div');
        cell.className = 'wg-cell ps-rel';
        inputGrid.appendChild(cell);
        inputCells.push(cell);
      }
    }
    var frame = document.createElement('div');
    frame.className = 'ps-frame';
    var inhost = q('inhost');
    inhost.appendChild(inputGrid);
    inhost.appendChild(frame);

    // ---- 输出 / dY 4×4 ----
    var outGrid = q('outhost');
    var outCells = [];
    for (var oi = 0; oi < 16; oi++) {
      var oc = document.createElement('div');
      oc.className = 'wg-cell';
      outGrid.appendChild(oc);
      outCells.push(oc);
    }

    function outValue(wr, wc) {
      var cells = windowCells(wr, wc);
      return state.mode === 'max' ? argmaxCell(cells).v : avgOf(cells);
    }

    function markFresh(cellEl) {
      if (freshCell && freshCell !== cellEl) { freshCell.classList.remove('ps-fresh'); }
      freshCell = cellEl;
      cellEl.classList.remove('ps-fresh');
      void cellEl.offsetWidth;  // 重启过渡，重复填充同一格也有动画
      cellEl.classList.add('ps-fresh');
    }

    function fillOutput(wr, wc, fresh) {
      var v = outValue(wr, wc);
      var el = outCells[wr * 4 + wc];
      el.textContent = fmt(v);
      el.style.background = inputColor(v);
      el.style.color = v >= 6 ? CARD : INK;
      state.computed[wr * 4 + wc] = true;
      if (fresh) { markFresh(el); }
      updateCount();
    }

    function updateCount() {
      var n = 0;
      state.computed.forEach(function (b) { if (b) { n += 1; } });
      q('count').textContent = state.view === 'fwd' ? ('已算 ' + n + '/16') : (state.backAll ? 'dY 全 1' : '点击选一格');
    }

    function paintOutput() {
      if (freshCell) { freshCell.classList.remove('ps-fresh'); freshCell = null; }
      if (state.view === 'fwd') {
        for (var n = 0; n < 16; n++) {
          var el = outCells[n];
          if (state.computed[n]) {
            var v = outValue(Math.floor(n / 4), n % 4);
            el.textContent = fmt(v);
            el.style.background = inputColor(v);
            el.style.color = v >= 6 ? CARD : INK;
          } else {
            el.textContent = '';
            el.style.background = CARD;
            el.style.color = INK;
          }
        }
      } else {
        for (var m = 0; m < 16; m++) {
          var on = state.backAll || m === state.wr * 4 + state.wc;
          var e2 = outCells[m];
          e2.textContent = on ? '1' : '0';
          e2.style.background = on ? INK : PAPER;
          e2.style.color = on ? '#c8ff47' : '#b9b2a4';
        }
      }
      updateCount();
    }

    function paintInput() {
      var cells = windowCells(state.wr, state.wc);
      var winSet = {}, argIdx = -1;
      cells.forEach(function (c) { winSet[c.r * 8 + c.c] = c; });
      if (state.mode === 'max') { argIdx = (function () { var a = argmaxCell(cells); return a.r * 8 + a.c; })(); }

      var route = null;
      if (state.view === 'bwd') {
        route = {};
        if (state.backAll) {
          for (var wr = 0; wr < 4; wr++) {
            for (var wc = 0; wc < 4; wc++) {
              var rm = routesFor(wr, wc, state.mode);
              for (var k in rm) { route[k] = rm[k]; }
            }
          }
        } else {
          route = routesFor(state.wr, state.wc, state.mode);
        }
      }

      for (var idx = 0; idx < 64; idx++) {
        var el = inputCells[idx];
        var r = Math.floor(idx / 8), c = idx % 8, v = INPUT[r][c];
        var oldBadge = el.querySelector('.ps-badge');
        if (oldBadge) { oldBadge.parentNode.removeChild(oldBadge); }
        el.classList.remove('ps-win-max', 'ps-win-avg', 'ps-dim');
        el.textContent = String(v);
        el.style.background = inputColor(v);
        el.style.color = v >= 6 ? CARD : INK;

        if (route) {
          var amt = route[idx];
          if (amt !== undefined) {
            el.style.background = 'rgba(192,57,43,' + (amt >= 1 ? 0.62 : 0.26) + ')';
            el.style.color = amt >= 1 ? CARD : INK;
            var b = document.createElement('span');
            b.className = 'ps-badge ps-bamt';
            b.textContent = fmt(amt);
            el.appendChild(b);
          } else {
            el.classList.add('ps-dim');
          }
        } else if (idx in winSet) {
          if (state.mode === 'max' && idx === argIdx) {
            el.classList.add('ps-win-max');
          } else if (state.mode === 'avg') {
            el.classList.add('ps-win-avg');
            // 均分：四格统一着色，表达"每格等权 1/4"，不再按数值深浅区分。
            el.style.background = 'rgba(11,99,243,.30)';
            el.style.color = INK;
            var bq = document.createElement('span');
            bq.className = 'ps-badge ps-bq';
            bq.textContent = '¼';
            el.appendChild(bq);
          }
        }
      }
    }

    function paintReadouts() {
      var cells = windowCells(state.wr, state.wc);
      var vals = cells.map(function (c) { return c.v; });
      var fwd = q('fwdread'), back = q('backread');

      if (state.view === 'fwd') {
        fwd.style.display = '';
        q('backpanel').style.display = 'none';
        if (state.mode === 'max') {
          var mi = maxInfo(cells), a = argmaxCell(cells);
          var html = '窗口 (' + state.wr + ',' + state.wc + ') · max(' + vals.join(', ') + ') = <b>' + mi.max + '</b> · 梯度只回 (' + a.r + ',' + a.c + ')';
          if (mi.ties > 1) { html += ' <span class="ps-warn">并列 ×' + mi.ties + ' → 行优先取第一个</span>'; }
          fwd.innerHTML = html;
        } else {
          fwd.innerHTML = '窗口 (' + state.wr + ',' + state.wc + ') · (' + vals.join('+') + ') / 4 = <b>' + fmt(avgOf(cells)) + '</b> · 四格各贡献 ¼';
        }
        return;
      }

      fwd.style.display = 'none';
      q('backpanel').style.display = '';
      var sel = 'dY[' + state.wr + ',' + state.wc + '] = 1';
      if (state.backAll) {
        if (state.mode === 'max') {
          back.innerHTML = 'dY 全 1 · Max 反向：只有 16 个胜出格各收到 <b>1</b>，其余 48 格 = <b>0</b>';
        } else {
          back.innerHTML = 'dY 全 1 · Avg 反向：64 格每格收到 <b>0.25</b>（窗口不重叠，每格只属一个窗口）';
        }
      } else if (state.mode === 'max') {
        var a2 = argmaxCell(cells), mi2 = maxInfo(cells);
        var h2 = sel + ' · Max 反向：dX[' + a2.r + ',' + a2.c + '] = <b>1</b>，其余 63 格 = <b>0</b>';
        if (mi2.ties > 1) { h2 += ' <span class="ps-warn">并列 ×' + mi2.ties + ' → 行优先取 (' + a2.r + ',' + a2.c + ')</span>'; }
        back.innerHTML = h2;
      } else {
        var coords = cells.map(function (c) { return '(' + c.r + ',' + c.c + ')'; });
        back.innerHTML = sel + ' · Avg 反向：' + coords.join(' ') + ' 各 = <b>0.25</b>';
      }
    }

    function paintLabels() {
      q('mode-max').classList.toggle('is-on', state.mode === 'max');
      q('mode-avg').classList.toggle('is-on', state.mode === 'avg');
      q('outlabel').textContent = state.view === 'fwd' ? '输出 Y (1,1,4,4)' : '上游梯度 dY (1,1,4,4)';
      q('pos').textContent = state.view === 'fwd' ? ('窗口 (' + state.wr + ',' + state.wc + ')') : ('dY 格 (' + state.wr + ',' + state.wc + ')');
      q('legend').textContent = state.view === 'bwd'
        ? '红 = 收到梯度的输入格'
        : (state.mode === 'max' ? '绿框 = 最大值胜出格（反向只回它）' : '虚线 = 四格均分（各 ¼）');
      q('sidenote').textContent = state.view === 'fwd'
        ? (state.mode === 'max' ? '每格 = 窗口内最大值。' : '每格 = 四格平均值。')
        : '点击格子选定 dY=1 的位置，或「dY 全设 1」。';
      q('view').textContent = state.view === 'fwd' ? '反向视图' : '回到正向';
      q('view').classList.toggle('is-primary', state.view === 'bwd');
      q('auto').classList.toggle('is-primary', state.view === 'fwd');
      q('all').textContent = state.view === 'fwd' ? '显示全部' : 'dY 全设 1';
      updateCount();
    }

    function paintAll() {
      paintLabels();
      paintInput();
      paintOutput();
      paintReadouts();
      positionFrame();
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
      var cellEl = inputCells[(state.wr * 2) * 8 + state.wc * 2].getBoundingClientRect();
      var gapX = pitch.x - pitch.w, gapY = pitch.y - pitch.h;
      frame.style.left = (cellEl.left - host.left - 2) + 'px';
      frame.style.top = (cellEl.top - host.top - 2) + 'px';
      frame.style.width = (pitch.w * 2 + gapX + 4) + 'px';
      frame.style.height = (pitch.h * 2 + gapY + 4) + 'px';
    }

    function setCursor(wr, wc, fill) {
      state.wr = Math.max(0, Math.min(3, wr));
      state.wc = Math.max(0, Math.min(3, wc));
      state.scan = state.wr * 4 + state.wc;
      if (state.view === 'fwd') {
        if (fill) { fillOutput(state.wr, state.wc, true); }
      } else {
        state.backAll = false;
      }
      paintLabels();
      paintInput();
      paintReadouts();
      if (state.view === 'bwd') { paintOutput(); }
      positionFrame();
    }

    function windowFromEvent(e) {
      if (!pitch) { measure(); }
      var rect = inputGrid.getBoundingClientRect();
      var col = Math.floor((e.clientX - rect.left + 1.5) / pitch.x);
      var row = Math.floor((e.clientY - rect.top + 1.5) / pitch.y);
      return {
        wr: Math.max(0, Math.min(3, Math.floor(Math.min(Math.max(0, row), 7) / 2))),
        wc: Math.max(0, Math.min(3, Math.floor(Math.min(Math.max(0, col), 7) / 2))),
      };
    }

    inputGrid.addEventListener('pointerdown', function (e) {
      e.preventDefault();
      dragging = true;
      try { inputGrid.setPointerCapture(e.pointerId); } catch (err) { /* 旧浏览器忽略 */ }
      var p = windowFromEvent(e);
      setCursor(p.wr, p.wc, true);
    });
    inputGrid.addEventListener('pointermove', function (e) {
      if (!dragging) { return; }
      var p = windowFromEvent(e);
      if (p.wr !== state.wr || p.wc !== state.wc) { setCursor(p.wr, p.wc, true); }
    });
    function endDrag() { dragging = false; }
    inputGrid.addEventListener('pointerup', endDrag);
    inputGrid.addEventListener('pointercancel', endDrag);

    outGrid.addEventListener('click', function (e) {
      var n = outCells.indexOf(e.target);
      if (n < 0) { return; }
      stopAuto();
      setCursor(Math.floor(n / 4), n % 4, false);
    });

    function stopAuto() {
      if (state.timer) { window.clearInterval(state.timer); state.timer = null; }
      q('auto').textContent = '自动扫描';
    }

    q('mode-max').addEventListener('click', function () {
      if (state.mode === 'max') { return; }
      stopAuto();
      state.mode = 'max';
      paintAll();
    });
    q('mode-avg').addEventListener('click', function () {
      if (state.mode === 'avg') { return; }
      stopAuto();
      state.mode = 'avg';
      paintAll();
    });

    q('view').addEventListener('click', function () {
      stopAuto();
      state.view = state.view === 'fwd' ? 'bwd' : 'fwd';
      paintAll();
    });

    q('auto').addEventListener('click', function () {
      if (state.timer) { stopAuto(); return; }
      if (state.scan >= 15) { state.scan = -1; }
      this.textContent = '暂停';
      state.timer = window.setInterval(function () {
        if (state.scan >= 15) { stopAuto(); return; }
        var n = state.scan + 1;
        setCursor(Math.floor(n / 4), n % 4, true);
      }, 320);
    });

    q('all').addEventListener('click', function () {
      stopAuto();
      if (state.view === 'fwd') {
        for (var r = 0; r < 4; r++) {
          for (var c = 0; c < 4; c++) { state.computed[r * 4 + c] = true; }
        }
        paintOutput();
      } else {
        state.backAll = true;
        paintAll();
      }
    });

    q('reset').addEventListener('click', function () {
      stopAuto();
      state.mode = 'max';
      state.view = 'fwd';
      state.backAll = false;
      state.computed = [];
      paintAll();
      fillOutput(0, 0, true);
      setCursor(1, 2, true);
    });

    var onResize = function () { pitch = null; positionFrame(); };
    window.addEventListener('resize', onResize);

    // 初始：预填 (0,0) 与 (1,2)（亮斑 + 两处 9 并列的窗口），窗口停在 (1,2)。
    paintAll();
    fillOutput(0, 0, true);
    setCursor(1, 2, true);

    return function dispose() {
      stopAuto();
      window.removeEventListener('resize', onResize);
      if (styleEl.parentNode) { styleEl.parentNode.removeChild(styleEl); }
    };
  }

  window.DLWidget.register('pool-sweep', mount);
})();
