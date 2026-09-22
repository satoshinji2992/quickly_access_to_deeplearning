/* next-token — 对照 shifted_labels 图的交互版：
 * 目标取输入的后一个 token；causal mask 管“不看未来”，
 * attention mask 管“不读 PAD key”，loss mask 管“PAD 目标不计损失”。
 * 点任意输入位置选中该 query；开关 PAD 观察两条屏蔽路径的差异。
 */
(function () {
  'use strict';

  var TOK = ['<BOS>', '我', '爱', '自然', '语言', '<EOS>', '<PAD>'];
  var N_REAL = 6;                       // 补 PAD 前的真实长度（含 EOS）
  var TARGET = ['我', '爱', '自然', '语言', '<EOS>', '−100', '−100'];
  /* 教学用预设：给定前文时下一个 token 的概率（非真实模型输出）。 */
  var P_NEXT = [0.82, 0.71, 0.63, 0.58, 0.44];

  var MONO = 'ui-monospace, SFMono-Regular, Consolas, monospace';
  var CSS = '' +
    '.nt-shift{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;border:1px dashed rgba(7,19,33,.35);background:#fffef9;padding:7px 10px;margin:0 0 12px;font:800 11px ' + MONO + ';color:#071321}' +
    '.nt-shift i{font-style:normal;color:#65717c;font-size:9px;letter-spacing:.08em}' +
    '.nt-shift b{color:#0b63f3}' +
    '.nt-sarr{font:700 8.5px ' + MONO + ';color:#0b63f3;white-space:nowrap}' +
    '.nt-legend{display:flex;flex-wrap:wrap;gap:8px 14px;margin:10px 0 12px;font:700 9px ' + MONO + ';color:#65717c;align-items:center}' +
    '.nt-chip{display:inline-block;width:14px;height:10px;border:1px solid rgba(7,19,33,.4);margin-right:4px;vertical-align:-1px;background:rgba(11,99,243,.5)}' +
    '.nt-chip.pad{background:repeating-linear-gradient(45deg,#f3d9d3 0 3px,#e5b8ac 3px 6px)}' +
    '.nt-chip.fut{background:#f4f2ec}' +
    '.nt-grid{display:grid;gap:3px;margin-bottom:6px}' +
    '.nt-lab{display:flex;flex-direction:column;justify-content:center;border:1px solid rgba(7,19,33,.25);background:#f4f2ec;padding:2px 6px;font:800 9px ' + MONO + ';color:#45515c;line-height:1.35}' +
    '.nt-lab span{font:700 8px ' + MONO + ';color:#65717c}' +
    '.nt-hd{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1px;border:1px solid rgba(7,19,33,.2);background:#f4f2ec;padding:3px 0}' +
    '.nt-hd i{font:700 8px ' + MONO + ';color:#65717c;font-style:normal}' +
    '.nt-hd b{font:800 10.5px ' + MONO + ';color:#071321}' +
    '.nt-cell{display:grid;place-items:center;border:1px solid rgba(7,19,33,.28);background:#fffef9;font:700 11px ' + MONO + ';color:#071321;min-height:34px;padding:2px 1px;text-align:center}' +
    'button.nt-cell{cursor:pointer;font:inherit}' +
    'button.nt-cell:hover{border-color:#071321}' +
    '.nt-cell.on{background:#071321;color:#c8ff47;border-color:#071321;box-shadow:2px 2px 0 #0b63f3}' +
    '.nt-cell.neg{background:#fdf3ee;color:#c0392b;border-color:#e5b8ac}' +
    '.nt-cell.zero{background:#f4f2ec;color:#9aa4ad}' +
    '.nt-cell.one{font-weight:800;color:#0b63f3}' +
    '.nt-cell.eos{color:#0b63f3}' +
    '.nt-matrix{display:grid;gap:2px;margin-top:4px}' +
    '.nt-corner{display:grid;place-items:center;border:1px solid rgba(7,19,33,.2);background:#f4f2ec;font:700 8px ' + MONO + ';color:#65717c}' +
    '.nt-kh{display:flex;align-items:center;justify-content:center;border:1px solid rgba(7,19,33,.2);background:#f4f2ec;font:700 9px ' + MONO + ';color:#45515c;min-height:22px;padding:0 1px;text-align:center}' +
    '.nt-kh.pad{color:#c0392b}' +
    '.nt-rh{display:flex;align-items:center;justify-content:flex-start;border:1px solid rgba(7,19,33,.25);background:#f4f2ec;font:700 9.5px ' + MONO + ';color:#071321;padding:0 5px;min-height:22px;white-space:nowrap;overflow:hidden}' +
    '.nt-mc{display:grid;place-items:center;border:1px solid rgba(7,19,33,.18);background:#fffef9;min-height:22px;font:700 9px ' + MONO + ';color:#b3bcc4}' +
    '.nt-mc.vis{background:rgba(11,99,243,.5);border-color:rgba(7,19,33,.28)}' +
    '.nt-mc.padk{background:repeating-linear-gradient(45deg,#f3d9d3 0 3px,#e5b8ac 3px 6px)}' +
    '.nt-mc.cur{border-color:#071321;box-shadow:2px 2px 0 #c8ff47}' +
    '.nt-mc.dim{opacity:.4}' +
    '.nt-flash{animation:nt-pop .5s ease-out both}' +
    '.nt-scroll{overflow-x:auto}' +
    '.nt-grid,.nt-matrix{min-width:432px}' +
    '@keyframes nt-pop{from{opacity:0}to{opacity:1}}';

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined) { node.textContent = text; }
    return node;
  }

  function mount(container) {
    container.innerHTML =
      '<style>' + CSS + '</style>' +
      '<p class="wg-title">Next-token 训练：每个位置预测后一个 token</p>' +
      '<p class="wg-sub">这里把「我爱自然语言」分成「我 / 爱 / 自然 / 语言」四个示意 token；训练脚本的字符词表会逐字切分。点击输入位置，可以同时查看它能读取的前文、预测目标和是否计算损失。</p>' +
      '<div class="wg-controls">' +
        '<button type="button" class="wg-button is-primary" data-role="padbtn">补 PAD 到等长：开</button>' +
        '<button type="button" class="wg-button" data-role="eosbtn">看 &lt;EOS&gt; 的两条路径</button>' +
      '</div>' +
      '<div class="nt-shift" data-role="shift"></div>' +
      '<div class="wg-label"><span>样本行（点击选择 query 位置）</span><span>对照 shifted_labels 图</span></div>' +
      '<div class="nt-scroll"><div class="nt-grid" data-role="rows"></div></div>' +
      '<div class="wg-label"><span>causal × padding 组合后的允许矩阵</span><span>行 = query ↓ · 列 = key →</span></div>' +
      '<div class="nt-legend">' +
        '<span><i class="nt-chip"></i>可读取（j ≤ t 且 key 有效）</span>' +
        '<span><i class="nt-chip fut"></i>未来（causal 禁止）</span>' +
        '<span><i class="nt-chip pad"></i>PAD key（attention 屏蔽）</span>' +
      '</div>' +
      '<div class="nt-scroll"><div class="nt-matrix" data-role="matrix"></div></div>' +
      '<div class="wg-readout">' +
        '<div class="wg-stat"><span data-role="s1l"></span><b data-role="s1"></b></div>' +
        '<div class="wg-stat"><span data-role="s2l"></span><b data-role="s2"></b></div>' +
        '<div class="wg-stat"><span data-role="s3l"></span><b data-role="s3"></b></div>' +
        '<div class="wg-stat"><span data-role="s4l"></span><b data-role="s4"></b></div>' +
      '</div>' +
      '<p class="wg-note" data-role="note"></p>';

    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };
    var state = { pad: true, sel: 2, timer: null };

    var len = function () { return state.pad ? 7 : 5; };
    var tokAt = function (t) { return TOK[t]; };
    var inputTokens = function () {
      var out = [];
      for (var t = 0; t < len(); t += 1) { out.push(tokAt(t)); }
      return out;
    };
    var targetTokens = function () {
      var out = [];
      for (var t = 0; t < len(); t += 1) {
        out.push(t <= 4 ? TARGET[t] : '−100');
      }
      return out;
    };
    var lossMask = function (t) { return t <= 4 ? 1 : 0; };
    var attnMask = function (t) { return (state.pad && t === 6) ? 0 : 1; };

    /* ---------- 样本行 ---------- */
    var rows = q('rows');
    // rows/matrix 都包在 .nt-scroll 里，窄屏可横向滚动
    var rowEls = {};   // name -> { labs:[], cells:[] }

    function buildRowHead() {
      var T = len();
      var hd = el('div');
      hd.className = 'nt-grid';
      hd.style.gridTemplateColumns = '64px repeat(' + T + ',1fr)';
      hd.appendChild(el('div', 'nt-corner', 'pos ╲'));
      for (var t = 0; t < T; t += 1) {
        var h = el('div', 'nt-hd');
        h.appendChild(el('i', '', String(t)));
        h.appendChild(el('b', '', tokAt(t)));
        hd.appendChild(h);
      }
      return hd;
    }

    function dataRow(name, label, sub, cellsBuilder) {
      var T = len();
      var g = el('div');
      g.className = 'nt-grid';
      g.style.gridTemplateColumns = '64px repeat(' + T + ',1fr)';
      var lab = el('div', 'nt-lab');
      lab.appendChild(el('b', '', label));
      lab.appendChild(el('span', '', sub));
      g.appendChild(lab);
      var cells = [];
      for (var t = 0; t < T; t += 1) {
        var c = cellsBuilder(t);
        g.appendChild(c);
        cells.push(c);
      }
      rows.appendChild(g);
      rowEls[name] = { g: g, lab: lab, cells: cells };
    }

    function buildRows() {
      rows.innerHTML = '';
      rowEls = {};
      rows.appendChild(buildRowHead());
      dataRow('x', '输入 x[t]', 'query 所在行', function (t) {
        var b = el('button', 'nt-cell', inputTokens()[t]);
        b.type = 'button';
        b.addEventListener('click', function () { state.sel = t; render(); });
        return b;
      });
      dataRow('y', '目标 y[t]', 'y[t] = x[t+1]', function (t) {
        var pad = t > 4;
        return el('div', 'nt-cell' + (pad ? ' neg' : (t === 4 ? ' eos' : '')), targetTokens()[t]);
      });
      dataRow('loss', 'loss mask', '目标不计损失为 0', function (t) {
        var m = lossMask(t);
        return el('div', 'nt-cell ' + (m ? 'one' : 'zero'), String(m));
      });
      dataRow('attn', 'attention mask', '0 = 该 key 禁止读取', function (t) {
        var m = attnMask(t);
        return el('div', 'nt-cell ' + (m ? 'one' : 'zero'), String(m));
      });
    }

    /* ---------- 允许矩阵 ---------- */
    var matrix = q('matrix');
    var matrixRows = [];   // { rh, cells:[{el, j}] }

    function buildMatrix() {
      matrix.innerHTML = '';
      matrixRows = [];
      var T = len();
      matrix.style.gridTemplateColumns = '64px repeat(' + T + ',1fr)';
      var hd = el('div');
      hd.className = 'nt-corner';
      matrix.appendChild(hd);
      for (var j = 0; j < T; j += 1) {
        matrix.appendChild(el('div', 'nt-kh' + (attnMask(j) ? '' : ' pad'), tokAt(j)));
      }
      for (var i = 0; i < T; i += 1) {
        matrix.appendChild(el('div', 'nt-rh', tokAt(i)));
        var row = { cells: [] };
        for (var k = 0; k < T; k += 1) {
          var c = el('div', 'nt-mc');
          matrix.appendChild(c);
          row.cells.push(c);
        }
        matrixRows.push(row);
      }
    }

    /* ---------- 渲染 ---------- */
    function esc(text) {
      return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
    function renderShift() {
      var T = len();
      var xs = [], ys = [];
      for (var t = 0; t < T; t += 1) {
        xs.push('<b>' + esc(inputTokens()[t]) + '</b>');
        ys.push(targetTokens()[t] === '−100' ? '<i style="color:#c0392b">−100</i>' : '<b>' + esc(targetTokens()[t]) + '</b>');
      }
      q('shift').innerHTML =
        '<i>X =</i> ' + xs.join(' ') +
        ' <span class="nt-sarr">label 右移一位 →</span> ' +
        '<i>Y =</i> ' + ys.join(' ');
    }

    function render() {
      var T = len();
      if (state.sel >= T) { state.sel = T - 1; }
      renderShift();

      /* 输入/目标/mask 行 */
      var selT = state.sel;
      ['x', 'y', 'loss', 'attn'].forEach(function (name) {
        rowEls[name].cells.forEach(function (cell, t) {
          cell.classList.toggle('on', name === 'x' ? t === selT : false);
          if (name === 'y') {
            cell.classList.toggle('on', t === selT && lossMask(selT) === 1);
          }
        });
      });

      /* 允许矩阵：重建列头（PAD 开关改变列数） */
      buildMatrix();
      matrixRows.forEach(function (row, i) {
        row.cells.forEach(function (cell, j) {
          cell.className = 'nt-mc';
          cell.textContent = '';
          cell.title = '';
          if (j > i) {
            cell.className = 'nt-mc fut' + (i === selT ? ' cur' : '');
            cell.textContent = '✕';
            cell.title = 'j > t：key 在未来，causal mask 置 −∞，softmax 后权重为 0';
            return;
          }
          if (!attnMask(j)) {
            cell.className = 'nt-mc padk' + (i === selT ? ' cur' : '');
            cell.title = 'j ≤ t 但 key 是 PAD：attention mask 屏蔽这一列';
            return;
          }
          cell.className = 'nt-mc vis' + (i === selT ? ' cur' : '') + (i > selT ? ' dim' : '');
          cell.title = '位置 ' + i + ' 可读取 ' + tokAt(j) + (i > selT ? '（非当前行，调暗显示）' : '');
        });
      });

      /* 读数 */
      var valid = 0;
      for (var v = 0; v < T; v += 1) { valid += lossMask(v); }
      var blocked = 0;
      for (var b = 0; b < T; b += 1) { blocked += attnMask(b) ? 0 : 1; }
      var sum = 0;
      for (var s = 0; s < 5; s += 1) { sum += Math.log(P_NEXT[s]); }
      var mean = -sum / 5;

      if (lossMask(selT)) {
        var p = P_NEXT[selT];
        q('s1l').textContent = '位置 ' + selT + ' 的损失项';
        q('s1').textContent = '−log ' + p.toFixed(2) + ' = ' + (-Math.log(p)).toFixed(2);
      } else {
        q('s1l').textContent = '位置 ' + selT + ' 的损失项';
        q('s1').textContent = '不计（−100）';
      }
      q('s2l').textContent = '有效 target（loss 分母）';
      q('s2').textContent = valid + ' / ' + T;
      q('s3l').textContent = 'attention 屏蔽列';
      q('s3').textContent = blocked ? blocked + '（PAD key）' : '0';
      q('s4l').textContent = '本句平均 loss';
      q('s4').textContent = mean.toFixed(2);

      /* 说明 */
      var padBtn = q('padbtn');
      padBtn.textContent = state.pad ? '补 PAD 到等长：开' : '补 PAD 到等长：关';
      padBtn.classList.toggle('is-primary', state.pad);

      var lines = [];
      if (state.pad) {
        lines.push('补 PAD 后 T=7：EOS 的输入位置 attention mask 仍是 1 —— 后面的位置本可以读它；但它的目标已是 PAD（−100），loss mask 为 0，不产生损失。');
        lines.push('PAD 这一行/列走的是另一条路径：作为 key 整列被 attention 屏蔽，作为 target 被 loss 忽略。causal 管时间方向，padding 管填充值，两者不能互相替代。');
      } else {
        lines.push('不补 PAD 时 T=5：所有位置都有效，两行 mask 全为 1，平均 loss 只除以 5。对照左边打开 PAD 的样子：补齐只是对齐 batch，不新增任何监督信号。');
      }
      lines.push('若忘记屏蔽，分母会错算成 ' + T + '，PAD 位置的 −log p 也会混进总和 —— 这正是 loss mask 要防的错。');
      q('note').textContent = lines.join(' ');
    }

    /* ---------- 事件 ---------- */
    q('padbtn').addEventListener('click', function () {
      state.pad = !state.pad;
      buildRows();
      render();
    });
    q('eosbtn').addEventListener('click', function () {
      state.pad = true;
      state.sel = 5;
      buildRows();
      render();
      var rowsEl = rowEls.x.cells[5];
      if (rowsEl) {
        rowsEl.classList.remove('nt-flash');
        void rowsEl.offsetWidth;
        rowsEl.classList.add('nt-flash');
      }
    });

    buildRows();
    render();
    return function dispose() {
      if (state.timer) { window.clearInterval(state.timer); }
    };
  }

  window.DLWidget.register('next-token', mount);
})();
