/* causal-mask — 同一个 6×6 因果下三角的两种用法：
 * 训练：一个 forward 并行算完全部行（label 右移一位）；
 * 推理：逐 token 步进，每步只算一行，上方行进缓存、下方行未到。
 */
(function () {
  'use strict';

  var TOK = ['BOS', '小', '猫', '在', '睡', '觉'];
  var T = TOK.length;   // 6

  /* 手工设定的 score（= q·k/√d 缩放后）。第 i 行 = 第 i 个 token 作为 query
     对各 key 的打分；上三角不存在 —— 会被因果掩码置 −∞。 */
  var SCORE = [
    [0.0],
    [0.2, 1.4],
    [0.1, 2.2, 1.1],
    [0.5, 1.0, 1.9, 0.3],
    [0.2, 0.6, 2.6, 2.0, 0.9],
    [0.1, 0.3, 2.4, 1.2, 2.8, 1.0],
  ];

  function softmax(row) {
    var max = row[0], i, sum = 0, es = [];
    for (i = 1; i < row.length; i += 1) { if (row[i] > max) { max = row[i]; } }
    for (i = 0; i < row.length; i += 1) { es.push(Math.exp(row[i] - max)); sum += es[i]; }
    return es.map(function (e) { return e / sum; });
  }

  var W = SCORE.map(softmax);
  var LOWER = T * (T + 1) / 2;    // 21 个可见 score
  var MASKED = T * T - LOWER;     // 15 个被掩码格

  var MONO = 'ui-monospace, SFMono-Regular, Consolas, monospace';
  var CSS = '' +
    '.cml-legend{display:flex;flex-wrap:wrap;gap:10px 14px;margin:10px 0 12px;font:700 9px ' + MONO + ';color:#65717c;align-items:center}' +
    '.cml-chip{display:inline-block;width:14px;height:10px;border:1px solid rgba(7,19,33,.4);margin-right:4px;vertical-align:-1px;background:rgba(11,99,243,.55)}' +
    '.cml-chip.mk{background:repeating-linear-gradient(45deg,#ece8da 0 3px,#d9d4c3 3px 6px)}' +
    '.cml-chip.nw{background:#c8ff47}' +
    '.cml-shift{display:flex;flex-wrap:wrap;align-items:center;gap:6px 12px;border:1px dashed rgba(7,19,33,.35);background:#fffef9;padding:7px 10px;margin:0 0 12px;font:800 11px ' + MONO + ';color:#071321}' +
    '.cml-shift i{font-style:normal;color:#65717c;font-size:9px;letter-spacing:.08em}' +
    '.cml-shift b{color:#0b63f3}' +
    '.cml-sarr{font:700 8.5px ' + MONO + ';color:#0b63f3;white-space:nowrap}' +
    '.cml-g{display:grid;grid-template-columns:60px repeat(6,1fr) 104px;gap:3px;margin-bottom:3px}' +
    '.cml-g.cml-hd{margin-bottom:7px}' +
    '.cml-ch{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1px;border:1px solid rgba(7,19,33,.25);background:#f4f2ec;padding:2px 0}' +
    '.cml-ch i{font:700 8px ' + MONO + ';color:#65717c;font-style:normal}' +
    '.cml-ch b{font:800 11px ' + MONO + ';color:#071321}' +
    '.cml-corner{display:grid;place-items:center;border:1px solid rgba(7,19,33,.25);background:#f4f2ec;font:700 8.5px ' + MONO + ';color:#65717c}' +
    '.cml-chr{display:grid;place-items:center;border:1px solid rgba(7,19,33,.25);background:#f4f2ec;font:700 8px ' + MONO + ';color:#65717c;text-align:center;line-height:1.4;padding:2px 3px}' +
    '.cml-rh{display:flex;flex-direction:column;justify-content:center;gap:1px;border:1px solid rgba(7,19,33,.3);background:#f4f2ec;padding:2px 6px;line-height:1.3}' +
    '.cml-rh b{font:800 12px ' + MONO + ';color:#071321}' +
    '.cml-rh span{font:700 8px ' + MONO + ';color:#65717c}' +
    '.cml-rh em{font:700 8px ' + MONO + ';font-style:normal;color:#0b63f3}' +
    '.cml-cell{height:44px;display:grid;place-items:center;border:1px solid rgba(7,19,33,.3);background:#fffef9;font:700 11px ' + MONO + ';color:#071321}' +
    '.cml-cell.mk{background:repeating-linear-gradient(45deg,#ede9dc 0 4px,#dad5c4 4px 8px);color:rgba(192,57,43,.8);font-size:13px}' +
    '.cml-cell.dim{opacity:.38}' +
    '.cml-cell.todo{border-style:dashed;background:transparent}' +
    '.cml-rbar{display:flex;gap:1px;border:1px solid rgba(7,19,33,.3);background:#fffef9;padding:2px}' +
    '.cml-rseg{min-width:0}' +
    '.cml-tag{display:grid;place-items:center;border:1px dashed rgba(7,19,33,.3);font:800 8.5px ' + MONO + ';letter-spacing:.05em;color:#9aa4ad}' +
    '.cml-tag.cache{border-style:solid;background:#f4f2ec;color:#65717c}' +
    '.cml-tag.now{border:1px solid #071321;background:#c8ff47;color:#071321}' +
    '.cml-matrix.step .cml-g.cur .cml-rh{background:#071321;border-color:#071321}' +
    '.cml-matrix.step .cml-g.cur .cml-rh b{color:#c8ff47}' +
    '.cml-matrix.step .cml-g.cur .cml-rh span{color:rgba(255,254,249,.75)}' +
    '.cml-matrix.step .cml-g.cur .cml-cell:not(.todo):not(.mk){border-color:#071321;box-shadow:2px 2px 0 #c8ff47}' +
    '.cml-matrix.train .cml-g:not(.cml-hd):hover .cml-cell{border-color:#071321}' +
    '.cml-matrix.train .cml-g:not(.cml-hd):hover .cml-rh b{color:#0b63f3}' +
    '.cml-bgrid{display:grid;grid-template-columns:60px repeat(6,1fr) 104px;gap:3px;align-items:end}' +
    '.cml-blab{grid-column:1;grid-row:1/3;display:grid;place-items:center;min-height:32px;border:1px solid rgba(7,19,33,.2);background:#f4f2ec;font:800 9px ' + MONO + ';color:#65717c}' +
    '.cml-bnum{height:13px;text-align:center;font:800 9px ' + MONO + ';color:#071321}' +
    '.cml-btrack{height:16px;border:1px solid rgba(7,19,33,.3);background:#fffef9}' +
    '.cml-btrack.todo{border-style:dashed}' +
    '.cml-bfill{height:100%;background:#0b63f3}' +
    '.cml-bnote{grid-column:8;grid-row:1/3;display:grid;place-items:center;padding:2px 4px;border:1px dashed rgba(7,19,33,.25);font:700 8px ' + MONO + ';color:#65717c;text-align:center;line-height:1.5}' +
    '@keyframes cml-pop{from{opacity:0}to{opacity:1}}' +
    '.cml-matrix.flash .cml-g:not(.cml-hd)>*{animation:cml-pop .45s ease-out both}';

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined) { node.textContent = text; }
    return node;
  }

  function alphaOf(w, max) {
    return (0.10 + 0.62 * (max > 0 ? w / max : 1)).toFixed(3);
  }

  function mount(container) {
    container.innerHTML =
      '<style>' + CSS + '</style>' +
      '<p class="wg-title">因果掩码：训练一次全算，推理每步只算一行</p>' +
      '<p class="wg-sub">T=6 · 下三角可见，其余置 −∞ · shape (1,1,6,6)</p>' +
      '<div class="wg-controls">' +
        '<button type="button" class="wg-button is-primary" data-role="mtrain">训练：一次 forward 全算</button>' +
        '<button type="button" class="wg-button" data-role="mstep">推理：逐 token 一步步算</button>' +
      '</div>' +
      '<div class="cml-legend">' +
        '<span><i class="cml-chip"></i>可见 score（越深 = softmax 权重越大）</span>' +
        '<span><i class="cml-chip mk"></i>被掩码 ✕ = −∞</span>' +
        '<span><i class="cml-chip nw"></i>本步新算（推理模式）</span>' +
      '</div>' +
      '<div class="cml-shift" data-role="shift">' +
        '<i>X =</i> BOS 小 猫 在 睡 觉' +
        '<span class="cml-sarr">label 右移一位 →</span>' +
        '<i>Y =</i> 小 猫 在 睡 觉 <b>EOS</b>' +
      '</div>' +
      '<div class="wg-label"><span data-role="mlab">score 矩阵 S = qKᵀ/√d</span><span>行 = query ↓ · 列 = key →</span></div>' +
      '<div class="cml-matrix train" data-role="matrix"></div>' +
      '<div data-role="bwrap" style="display:none">' +
        '<div class="wg-label"><span data-role="blab"></span><span>softmax 只在可见 key 上归一</span></div>' +
        '<div class="cml-bgrid" data-role="bgrid"></div>' +
      '</div>' +
      '<div class="wg-readout">' +
        '<div class="wg-stat"><span data-role="s1l"></span><b data-role="s1"></b></div>' +
        '<div class="wg-stat"><span data-role="s2l"></span><b data-role="s2"></b></div>' +
        '<div class="wg-stat"><span data-role="s3l"></span><b data-role="s3"></b></div>' +
        '<div class="wg-stat"><span data-role="s4l"></span><b data-role="s4"></b></div>' +
      '</div>' +
      '<div class="wg-controls" data-role="ctrain">' +
        '<button type="button" class="wg-button is-primary" data-role="replay">重放一次 forward（6 行同时点亮）</button>' +
      '</div>' +
      '<div class="wg-controls" data-role="cstep" style="display:none">' +
        '<button type="button" class="wg-button is-primary" data-role="next">算下一行</button>' +
        '<button type="button" class="wg-button" data-role="auto">自动播放</button>' +
        '<button type="button" class="wg-button" data-role="reset">重置</button>' +
      '</div>' +
      '<p class="wg-note" data-role="note"></p>';

    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };
    var state = { mode: 'train', step: 0, timer: null };

    /* 矩阵：表头行 + 6 个数据行，网格模板一致保证列对齐。 */
    var matrix = q('matrix');
    var hd = el('div', 'cml-g cml-hd');
    hd.appendChild(el('div', 'cml-corner', 'q ╲ k'));
    var colHeads = [];
    for (var j = 0; j < T; j += 1) {
      var ch = el('div', 'cml-ch');
      ch.appendChild(el('i', '', String(j)));
      ch.appendChild(el('b', '', TOK[j]));
      hd.appendChild(ch);
      colHeads.push(ch);
    }
    var hdRight = el('div', 'cml-chr', '每行 softmax');
    hd.appendChild(hdRight);
    matrix.appendChild(hd);

    var rows = [];
    for (var i = 0; i < T; i += 1) {
      var g = el('div', 'cml-g');
      var rh = el('div', 'cml-rh');
      rh.appendChild(el('b', '', TOK[i]));
      rh.appendChild(el('span', '', 'pos ' + i));
      rh.appendChild(el('em', '', ''));
      g.appendChild(rh);
      var cells = [];
      for (var k = 0; k < T; k += 1) { cells.push(el('div', 'cml-cell')); g.appendChild(cells[cells.length - 1]); }
      var rbar = el('div', 'cml-rbar');
      var tag = el('div', 'cml-tag');
      g.appendChild(rbar);
      g.appendChild(tag);
      matrix.appendChild(g);
      rows.push({ g: g, rh: rh, tok: rh.children[0], pos: rh.children[1], pred: rh.children[2], cells: cells, rbar: rbar, tag: tag });
    }

    /* 推理模式：当前行 softmax 横条（与矩阵列对齐）。 */
    var bgrid = q('bgrid');
    bgrid.appendChild(el('div', 'cml-blab', 'w'));
    var bnums = [], btracks = [];
    for (var b = 0; b < T; b += 1) { bnums.push(el('div', 'cml-bnum')); bgrid.appendChild(bnums[b]); }
    for (var t2 = 0; t2 < T; t2 += 1) {
      var tr = el('div', 'cml-btrack');
      var fill = el('div', 'cml-bfill');
      tr.appendChild(fill);
      bgrid.appendChild(tr);
      btracks.push({ tr: tr, fill: fill });
    }
    bgrid.appendChild(el('div', 'cml-bnote', '条长 = w ÷ 行最大 w'));

    var btnTrain = q('mtrain'), btnStep = q('mstep'), btnNext = q('next'), btnAuto = q('auto'), btnReset = q('reset'), btnReplay = q('replay');

    function renderRowBar(row, i) {
      row.rbar.innerHTML = '';
      var max = Math.max.apply(null, W[i]);
      for (var j = 0; j <= i; j += 1) {
        var seg = el('div', 'cml-rseg');
        seg.style.flex = String(W[i][j]) + ' 1 0px';
        seg.style.background = 'rgba(11,99,243,' + alphaOf(W[i][j], max) + ')';
        seg.title = TOK[j] + '：' + W[i][j].toFixed(2);
        row.rbar.appendChild(seg);
      }
      row.rbar.title = '第 ' + i + ' 行 softmax：' + W[i].map(function (w) { return w.toFixed(2); }).join(' / ');
    }

    function render() {
      var train = state.mode === 'train';
      var step = state.step;
      matrix.className = 'cml-matrix ' + (train ? 'train' : 'step');
      q('shift').style.display = train ? '' : 'none';
      q('bwrap').style.display = train ? 'none' : '';
      q('ctrain').style.display = train ? '' : 'none';
      q('cstep').style.display = train ? 'none' : '';
      btnTrain.classList.toggle('is-primary', train);
      btnStep.classList.toggle('is-primary', !train);
      hdRight.textContent = train ? '每行 softmax' : '行状态';
      q('mlab').textContent = train ? 'score 矩阵 S = qKᵀ/√d（一次全算）' : 'score 矩阵 S = qKᵀ/√d（逐行计算）';

      rows.forEach(function (row, i) {
        var max = Math.max.apply(null, W[i]);
        row.tok.textContent = TOK[i];
        row.pos.textContent = 'pos ' + i;
        row.pred.style.display = train ? '' : 'none';
        if (train) { row.pred.textContent = i < T - 1 ? '预测 ' + TOK[i + 1] : '预测 EOS'; }
        row.rbar.style.display = train ? '' : 'none';
        row.tag.style.display = train ? 'none' : '';

        if (train) {
          row.g.className = 'cml-g';
          renderRowBar(row, i);
        } else {
          row.tag.className = 'cml-tag' + (i < step ? ' cache' : i === step ? ' now' : '');
          row.tag.textContent = i < step ? '缓存' : i === step ? '本步新算' : '未到';
          row.tag.title = i < step ? '已算过，进缓存' : i === step ? '本步新算的一行' : '还没轮到';
          row.g.className = i === step ? 'cml-g cur' : 'cml-g';
        }

        row.cells.forEach(function (cell, j) {
          cell.style.background = '';
          cell.style.boxShadow = '';
          if (!train && i > step) {
            /* 未到的行：整行虚线留空 —— 这些 score 还不存在。 */
            cell.className = 'cml-cell todo';
            cell.textContent = '';
            cell.title = '还未算到这一行';
            return;
          }
          if (j > i) {
            cell.className = 'cml-cell mk';
            cell.textContent = '✕';
            cell.title = 'j > i：被掩码置 −∞';
            return;
          }
          var s = SCORE[i][j], w = W[i][j];
          cell.textContent = s.toFixed(1);
          cell.style.background = 'rgba(11,99,243,' + alphaOf(w, max) + ')';
          cell.title = 'S[' + i + ',' + j + '] = ' + s.toFixed(1) + ' · softmax 权重 ' + w.toFixed(2) +
            (!train && i < step ? ' · 已缓存' : '') + (!train && i === step ? ' · 本步新算' : '');
          if (!train) {
            if (i < step) { cell.className = 'cml-cell dim'; }
            else { cell.className = 'cml-cell'; }
          } else {
            cell.className = 'cml-cell';
          }
        });
      });

      if (!train) {
        var wmax = Math.max.apply(null, W[step]);
        q('blab').textContent = '第 ' + (step + 1) + ' 步 · query「' + TOK[step] + '」行的 softmax';
        bnums.forEach(function (n, j) { n.textContent = j <= step ? W[step][j].toFixed(2) : ''; });
        btracks.forEach(function (bt, j) {
          bt.tr.classList.toggle('todo', j > step);
          bt.fill.style.width = j <= step ? (W[step][j] / wmax * 100).toFixed(1) + '%' : '0%';
        });
      }

      if (train) {
        q('s1l').textContent = '并行 score'; q('s1').textContent = LOWER + ' / ' + (T * T) + ' 格';
        q('s2l').textContent = '被掩码（−∞）'; q('s2').textContent = MASKED + ' 格';
        q('s3l').textContent = 'forward 次数'; q('s3').textContent = '1 次';
        q('s4l').textContent = '训练 label'; q('s4').textContent = '右移一位';
        q('note').textContent = '悬停任一行查看该行 softmax';
      } else {
        q('s1l').textContent = '当前步'; q('s1').textContent = (step + 1) + ' / ' + T;
        q('s2l').textContent = '本步新算 score'; q('s2').textContent = '1×' + (step + 1) + ' = ' + (step + 1) + ' 个';
        q('s3l').textContent = '已缓存行'; q('s3').textContent = step + ' 行';
        q('s4l').textContent = 'forward 次数'; q('s4').textContent = (step + 1) + ' 次';
        if (step < T - 1) {
          q('note').textContent = '第 ' + (step + 1) + ' / ' + T + ' 步：新算 pos ' + step + ' 一行，上方 ' + step + ' 行已缓存';
        } else {
          q('note').textContent = '完成：共 ' + T + ' 次 forward，K/V 从缓存读';
        }
        btnNext.disabled = step >= T - 1;
        btnAuto.disabled = step >= T - 1;
        btnNext.textContent = step >= T - 1 ? '已完成 ' + T + ' / ' + T + ' 行' : '算下一行（' + (step + 2) + ' / ' + T + '）';
      }
    }

    function stopAuto() {
      if (state.timer) {
        window.clearInterval(state.timer);
        state.timer = null;
      }
      btnAuto.textContent = '自动播放';
    }

    btnTrain.addEventListener('click', function () { stopAuto(); state.mode = 'train'; render(); });
    btnStep.addEventListener('click', function () { stopAuto(); state.mode = 'step'; render(); });
    btnReplay.addEventListener('click', function () {
      matrix.classList.remove('flash');
      void matrix.offsetWidth;
      matrix.classList.add('flash');
    });
    btnNext.addEventListener('click', function () {
      if (state.step < T - 1) { state.step += 1; render(); }
    });
    btnAuto.addEventListener('click', function () {
      if (state.timer) { stopAuto(); return; }
      if (state.step >= T - 1) { return; }
      btnAuto.textContent = '暂停';
      state.timer = window.setInterval(function () {
        if (state.step < T - 1) { state.step += 1; render(); }
        if (state.step >= T - 1) { stopAuto(); }
      }, 800);
    });
    btnReset.addEventListener('click', function () { stopAuto(); state.step = 0; render(); });

    render();
    return function dispose() { stopAuto(); };
  }

  window.DLWidget.register('causal-mask', mount);
})();
