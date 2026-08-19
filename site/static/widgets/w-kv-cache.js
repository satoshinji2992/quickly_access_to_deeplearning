/* kv-cache — prefill / decode 步进动画：看每层 K/V 缓存增长与 GQA 复用。
 *
 * 设定：prompt = BOS 小 猫 在 睡（5 个位置），decode 生成 觉、。（2 步）。
 * 两层（Layer 0 / Layer 1），每层 K/V 各 (1, 2, past_len, 8)；
 * Hq=4 个 query heads 经 repeat_kv 复用 Hkv=2 份 K/V。
 * 槽位 = 一个 token 位置，内部 2 个子格 = 2 个 KV heads。
 */
(function () {
  'use strict';

  var PROMPT = ['BOS', '小', '猫', '在', '睡'];
  var GEN = ['觉', '。'];
  var MAXPOS = PROMPT.length + GEN.length;   // 7 个位置，位置编号 0..6

  // 每层一排“softmax 权重”示意：手工设定的递增序列，渲染时取前 past 个并归一化。
  var W_BASE = [
    [0.05, 0.08, 0.12, 0.16, 0.22, 0.30, 0.38],
    [0.10, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20],
  ];

  var MONO = 'ui-monospace, SFMono-Regular, Consolas, monospace';
  var CSS = '' +
    '.kvc-legend{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0 12px;font:700 9px ' + MONO + ';color:#65717c;align-items:center}' +
    '.kvc-chip{display:inline-block;width:13px;height:9px;border:1px solid rgba(7,19,33,.4);margin-right:4px;vertical-align:-1px}' +
    '.kvc-chip.dash{border-style:dashed;background:transparent}' +
    '.kvc-row{display:flex;gap:4px;margin-bottom:4px}' +
    '.kvc-rl{flex:0 0 22px;font:700 9px ' + MONO + ';color:#65717c;padding-top:3px}' +
    '.kvc-cols{flex:1 1 auto;display:grid;grid-template-columns:repeat(7,1fr);gap:4px;min-width:0}' +
    '.kvc-tok{text-align:center;font:700 10px ' + MONO + ';color:#071321;padding:1px 0}' +
    '.kvc-tok.gen{color:#0b63f3}' +
    '.kvc-tok.ghost{opacity:.25}' +
    '.kvc-slot{display:flex;gap:2px;padding:2px;border:1px dashed rgba(7,19,33,.25)}' +
    '.kvc-slot.on{border-style:solid;border-color:rgba(7,19,33,.3);background:#fffef9}' +
    '.kvc-cell{flex:1 1 0;height:13px;transition:background-color .5s ease}' +
    '.kvc-slot.k .kvc-cell.f{background:#0b63f3}' +
    '.kvc-slot.v .kvc-cell.f{background:#72d7ff}' +
    '.kvc-slot.new .kvc-cell.f{background:#c8ff47}' +
    '.kvc-pos{text-align:center;font:700 9px ' + MONO + ';color:#65717c}' +
    '.kvc-pos.cur{color:#0b63f3}' +
    '.kvc-btrack{height:42px;border:1px solid rgba(7,19,33,.12);background:#fffef9;display:flex;align-items:flex-end;justify-content:center}' +
    '.kvc-bar{width:72%;height:0;background:#0b63f3;transition:height .35s ease,background-color .5s ease;display:flex;justify-content:center;align-items:flex-start}' +
    '.kvc-bar.new{background:#c8ff47}' +
    '.kvc-bar b{font:800 8px ' + MONO + ';color:#071321;line-height:1.2;margin-top:2px}' +
    '.kvc-layer{border:1px solid rgba(7,19,33,.25);background:#f4f2ec;padding:8px;margin-bottom:10px}' +
    '.kvc-lh{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px;font:800 10px ' + MONO + ';color:#071321}' +
    '.kvc-lh span{font:700 8.5px ' + MONO + ';color:#65717c}' +
    '.kvc-qbox{display:flex;align-items:center;flex-wrap:wrap;gap:6px;border:1px solid rgba(7,19,33,.3);background:#f4f2ec;padding:8px;margin-bottom:10px;font:700 10px ' + MONO + ';color:#45515c}' +
    '.kvc-qcap{color:#65717c}' +
    '.kvc-tchip{border:1px solid rgba(7,19,33,.35);background:#fffef9;padding:1px 4px;color:#071321}' +
    '.kvc-qtok{min-width:34px;text-align:center;border:1px solid #071321;background:#fffef9;padding:3px 6px;font:800 13px ' + MONO + ';color:#071321}' +
    '.kvc-qpos{color:#65717c}' +
    '.kvc-qheads{display:flex;gap:3px;padding:4px;border:1px dashed rgba(7,19,33,.35);background:#fffef9}' +
    '.kvc-qh{width:12px;height:12px;background:#0b63f3}' +
    '.kvc-mono{font:800 10px ' + MONO + ';color:#071321}' +
    '.kvc-out{color:#0b63f3}' +
    '.kvc-arr{color:#65717c}' +
    '.kvc-gqa{border:1px solid rgba(7,19,33,.25);background:#fffef9;padding:8px 10px;display:flex;flex-direction:column;gap:5px}' +
    '.kvc-grow{display:flex;align-items:center;gap:4px;font:700 9px ' + MONO + ';color:#45515c;flex-wrap:wrap}' +
    '.kvc-cq{border:1px solid #0b63f3;color:#0b63f3;background:#fffef9;padding:2px 4px}' +
    '.kvc-ck{background:#0b63f3;color:#fffef9;padding:2px 4px}' +
    '.kvc-cv{background:#72d7ff;color:#071321;padding:2px 4px}' +
    '.kvc-gtxt{font:700 9px ' + MONO + ';color:#65717c;line-height:1.6}' +
    '.kvc-controls .wg-button:disabled{opacity:.4;cursor:default}' +
    '.kvc-controls .wg-button:disabled:hover{background:#fffef9;color:#071321;box-shadow:none}' +
    '.kvc-controls .wg-button.is-primary:disabled:hover{background:#071321;color:#c8ff47}';

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined) { node.textContent = text; }
    return node;
  }

  function tokenAt(i) { return i < PROMPT.length ? PROMPT[i] : GEN[i - PROMPT.length]; }

  function weights(layer, n) {
    var base = W_BASE[layer], sum = 0, out = [], i;
    for (i = 0; i < n; i += 1) { sum += base[i]; }
    for (i = 0; i < n; i += 1) { out.push(base[i] / sum); }
    return out;
  }

  function buildLayer(l) {
    var block = el('div', 'kvc-layer');
    var lh = el('div', 'kvc-lh');
    lh.appendChild(el('b', '', 'Layer ' + l));
    var wmax = el('span', '', '');
    lh.appendChild(wmax);
    block.appendChild(lh);

    // attention 权重条：与 K 行同一套 7 列网格，对齐每个位置。
    var attRow = el('div', 'kvc-row');
    attRow.appendChild(el('div', 'kvc-rl', 'att'));
    var attCols = el('div', 'kvc-cols');
    var bars = [];
    for (var b = 0; b < MAXPOS; b += 1) {
      var track = el('div', 'kvc-btrack');
      var bar = el('div', 'kvc-bar');
      bar.appendChild(el('b', '', ''));
      track.appendChild(bar);
      attCols.appendChild(track);
      bars.push(bar);
    }
    attRow.appendChild(attCols);
    block.appendChild(attRow);

    var refs = { attRow: attRow, bars: bars, wmax: wmax, k: [], v: [], pos: [] };
    [['k', 'K'], ['v', 'V']].forEach(function (pair) {
      var row = el('div', 'kvc-row');
      row.appendChild(el('div', 'kvc-rl', pair[1]));
      var cols = el('div', 'kvc-cols');
      for (var i = 0; i < MAXPOS; i += 1) {
        var slot = el('div', 'kvc-slot ' + pair[0]);
        slot.appendChild(el('span', 'kvc-cell'));
        slot.appendChild(el('span', 'kvc-cell'));
        cols.appendChild(slot);
        refs[pair[0]].push(slot);
      }
      row.appendChild(cols);
      block.appendChild(row);
    });

    var posRow = el('div', 'kvc-row');
    posRow.appendChild(el('div', 'kvc-rl', 'pos'));
    var posCols = el('div', 'kvc-cols');
    for (var p = 0; p < MAXPOS; p += 1) {
      refs.pos.push(el('div', 'kvc-pos', String(p)));
      posCols.appendChild(refs.pos[refs.pos.length - 1]);
    }
    posRow.appendChild(posCols);
    block.appendChild(posRow);
    return { root: block, refs: refs };
  }

  function mount(container) {
    var style = document.createElement('style');
    style.id = 'widget-kvc-style';
    style.textContent = CSS;
    document.head.appendChild(style);

    container.innerHTML =
      '<p class="wg-title">KV Cache：prefill 与逐步 decode</p>' +
      '<p class="wg-sub">prompt = BOS 小 猫 在 睡，之后逐 token 生成 觉、。两层各存一份 K/V：Hkv=2、Hq=4、head_dim=8。每个槽位 = 一个 token 位置，内部 2 个子格 = 2 个 KV heads。</p>' +
      '<div class="kvc-legend">' +
        '<span><i class="kvc-chip" style="background:#0b63f3"></i>K ×2 heads</span>' +
        '<span><i class="kvc-chip" style="background:#72d7ff"></i>V ×2 heads</span>' +
        '<span><i class="kvc-chip" style="background:#c8ff47"></i>本步新写入</span>' +
        '<span><i class="kvc-chip dash"></i>空槽位</span>' +
      '</div>' +
      '<div class="kvc-row" data-role="tokrow"></div>' +
      '<div class="wg-label"><span>当前 query（decode 时）</span><span>RoPE 位置从 past_len 接续</span></div>' +
      '<div class="kvc-qbox" data-role="qbox"></div>' +
      '<div class="wg-label"><span>每层缓存</span><span>K 行 / V 行 × 7 个位置</span></div>' +
      '<div data-role="layers"></div>' +
      '<div class="wg-label"><span>状态</span><span>shape = (B, Hkv, past_len, Dh)</span></div>' +
      '<div class="wg-readout">' +
        '<div class="wg-stat"><span>past_len</span><b data-role="st-past"></b></div>' +
        '<div class="wg-stat"><span>K/V shape（每层）</span><b data-role="st-shape"></b></div>' +
        '<div class="wg-stat"><span>本步 QKᵀ（用缓存）</span><b data-role="st-qk"></b></div>' +
        '<div class="wg-stat"><span>朴素重算 QKᵀ</span><b data-role="st-naive"></b></div>' +
      '</div>' +
      '<div class="wg-label"><span>GQA 分组</span><span>Hq=4 / Hkv=2</span></div>' +
      '<div class="kvc-gqa">' +
        '<div class="kvc-grow"><span class="kvc-cq">q0</span><span class="kvc-cq">q1</span><span class="kvc-arr">─ 复用 →</span><span class="kvc-ck">K0</span><span class="kvc-cv">V0</span></div>' +
        '<div class="kvc-grow"><span class="kvc-cq">q2</span><span class="kvc-cq">q3</span><span class="kvc-arr">─ 复用 →</span><span class="kvc-ck">K1</span><span class="kvc-cv">V1</span></div>' +
        '<div class="kvc-gtxt">每层只缓存 2 份 K/V（Hkv=2），4 个 query heads 计算时按 repeat_kv 分组复用；槽位里的 2 个子格就是这 2 个 KV heads。</div>' +
      '</div>' +
      '<div class="wg-controls kvc-controls">' +
        '<button type="button" class="wg-button" data-role="pre">Prefill 5 个 prompt token</button>' +
        '<button type="button" class="wg-button" data-role="dec">Decode 一步</button>' +
        '<button type="button" class="wg-button" data-role="auto">自动解码</button>' +
        '<button type="button" class="wg-button" data-role="reset">重置</button>' +
      '</div>' +
      '<p class="wg-note" data-role="note"></p>';

    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };
    var state = { past: 0, phase: 'empty', busy: false, timer: null, fade: null, newPos: -1 };

    // token 行：左标签 tok + 7 个 token 格（生成 token 未写入前半透明）。
    var tokRow = q('tokrow');
    tokRow.appendChild(el('div', 'kvc-rl', 'tok'));
    var tokCols = el('div', 'kvc-cols');
    var tokCells = [];
    for (var t = 0; t < MAXPOS; t += 1) {
      var cell = el('div', 'kvc-tok');
      tokCols.appendChild(cell);
      tokCells.push(cell);
    }
    tokRow.appendChild(tokCols);

    var layers = [buildLayer(0), buildLayer(1)];
    layers.forEach(function (L) { q('layers').appendChild(L.root); });

    var btnPre = q('pre'), btnDec = q('dec'), btnAuto = q('auto'), btnReset = q('reset');

    function qHeads() {
      var s = '';
      for (var i = 0; i < 4; i += 1) { s += '<span class="kvc-qh"></span>'; }
      return '<span class="kvc-qheads">' + s + '</span>';
    }

    function renderQbox() {
      var box = q('qbox');
      if (state.phase === 'empty') { box.style.display = 'none'; return; }
      box.style.display = '';
      if (state.phase === 'prefilling' || state.phase === 'prefilled') {
        var chips = '';
        PROMPT.forEach(function (tk) { chips += '<span class="kvc-tchip">' + tk + '</span>'; });
        box.innerHTML = '<span class="kvc-qcap">输入</span>' + chips +
          '<span class="kvc-arr">→</span>' + qHeads() +
          '<span class="kvc-mono">q (1,4,5,8)</span>' +
          '<span>训练式并行：一次 forward 算完所有位置</span>';
        return;
      }
      var idx = state.past - 1;
      var next = state.past >= MAXPOS ? null : GEN[state.past - PROMPT.length];
      box.innerHTML = '<span class="kvc-qcap">输入</span>' +
        '<span class="kvc-qtok">' + tokenAt(idx) + '</span>' +
        '<span class="kvc-qpos">pos ' + idx + '</span>' +
        '<span class="kvc-arr">→</span>' + qHeads() +
        '<span class="kvc-mono">q (1,4,1,8)</span><span class="kvc-arr">→</span>' +
        (next ? '<span class="kvc-out">产出「' + next + '」</span>' : '<span class="kvc-out">序列完成</span>');
    }

    function noteText() {
      if (state.phase === 'empty') {
        return '缓存为空。Prefill 一次并行 forward 写入 5 个 prompt 位置；之后每步 decode 只 forward 1 个新 token。';
      }
      if (state.phase === 'prefilling') {
        return '训练式并行：一次 forward 算完所有位置（动画把 5 列拆开逐列点亮，只是便于观看）。';
      }
      if (state.phase === 'prefilled') {
        return 'past_len = 5。下一步 decode：新 token 的 q 只与缓存 K 做一次 1×5 的 QKᵀ，它自己的 K/V 追加为第 6 列。';
      }
      if (state.phase === 'decode') {
        return 'decode 第 ' + (state.past - PROMPT.length) + ' 步：q (1,4,1,8) 复用 2 份 K/V 做注意力。QKᵀ 只算 1×' +
          state.past + ' = ' + state.past + ' 个 score；朴素重算要 ' + state.past + '×' + state.past + ' = ' + state.past * state.past + ' 个。';
      }
      return '生成结束：BOS 小 猫 在 睡 觉 。两步 decode 共追加 2 列，缓存增长到 (1,2,7,8)；每一步都只新算 1 行 score。';
    }

    function render() {
      tokCells.forEach(function (c, i) {
        var isGen = i >= PROMPT.length;
        var written = i < state.past;
        c.textContent = tokenAt(i);
        c.classList.toggle('ghost', isGen && !written);
        c.classList.toggle('gen', isGen && written);
      });

      renderQbox();

      layers.forEach(function (L, l) {
        ['k', 'v'].forEach(function (kind) {
          L.refs[kind].forEach(function (slot, i) {
            var filled = i < state.past;
            slot.classList.toggle('on', filled);
            slot.classList.toggle('new', filled && i === state.newPos);
            slot.title = filled ? 'pos ' + i + ' · ' + tokenAt(i) + ' · 2 个 KV heads' : '空槽位';
            for (var c = 0; c < slot.children.length; c += 1) {
              slot.children[c].classList.toggle('f', filled);
            }
          });
        });
        L.refs.pos.forEach(function (p, i) { p.classList.toggle('cur', i === state.past - 1); });

        var showAtt = state.phase === 'decode' || state.phase === 'done';
        L.refs.attRow.style.display = showAtt ? '' : 'none';
        if (showAtt) {
          var ws = weights(l, state.past);
          var max = 0, at = 0;
          ws.forEach(function (w, i) { if (w > max) { max = w; at = i; } });
          L.refs.bars.forEach(function (bar, i) {
            bar.style.height = i < state.past ? Math.round(ws[i] / max * 100) + '%' : '0%';
            bar.classList.toggle('new', i === state.newPos);
            bar.firstChild.textContent = i === at ? ws[i].toFixed(2) : '';
          });
          L.refs.wmax.textContent = 'softmax(qKᵀ/√8) · 最大权重 ' + max.toFixed(2);
        } else {
          L.refs.wmax.textContent = state.phase === 'empty' ? '等待 prefill' : 'prefill：完整因果矩阵并行算';
        }
      });

      q('st-past').textContent = String(state.past);
      q('st-shape').textContent = '(1,2,' + state.past + ',8)';
      if (state.phase === 'empty') {
        q('st-qk').textContent = '—';
        q('st-naive').textContent = '—';
      } else if (state.phase === 'prefilling' || state.phase === 'prefilled') {
        q('st-qk').textContent = '5×5 = 25';
        q('st-naive').textContent = '5×5 = 25';
      } else {
        q('st-qk').textContent = '1×' + state.past + ' = ' + state.past;
        q('st-naive').textContent = state.past + '×' + state.past + ' = ' + state.past * state.past;
      }

      var canDecode = state.past >= PROMPT.length && state.past < MAXPOS;
      btnPre.disabled = state.busy || state.phase !== 'empty';
      btnDec.disabled = state.busy || !canDecode;
      btnAuto.disabled = state.busy || !canDecode;
      btnReset.disabled = state.busy;
      btnDec.textContent = canDecode ? 'Decode 一步（剩 ' + (MAXPOS - state.past) + ' 步）' : 'Decode 一步';
      btnPre.classList.toggle('is-primary', !state.busy && state.phase === 'empty');
      btnDec.classList.toggle('is-primary', !state.busy && canDecode);

      q('note').textContent = noteText();
    }

    function prefill() {
      if (state.busy || state.phase !== 'empty') { return; }
      state.busy = true;
      state.phase = 'prefilling';
      render();
      state.timer = window.setInterval(function () {
        state.past += 1;
        if (state.past >= PROMPT.length) {
          window.clearInterval(state.timer);
          state.timer = null;
          state.busy = false;
          state.phase = 'prefilled';
        }
        render();
      }, 150);
    }

    function decodeStep() {
      if (state.past < PROMPT.length || state.past >= MAXPOS) { return; }
      state.newPos = state.past;
      state.past += 1;
      state.phase = state.past >= MAXPOS ? 'done' : 'decode';
      if (state.fade) { window.clearTimeout(state.fade); }
      state.fade = window.setTimeout(function () {
        state.fade = null;
        state.newPos = -1;
        render();
      }, 950);
      render();
    }

    function autoDecode() {
      if (state.busy || state.past < PROMPT.length || state.past >= MAXPOS) { return; }
      state.busy = true;
      render();
      state.timer = window.setInterval(function () {
        decodeStep();
        if (state.past >= MAXPOS) {
          window.clearInterval(state.timer);
          state.timer = null;
          state.busy = false;
          render();
        }
      }, 850);
    }

    function reset() {
      if (state.timer) { window.clearInterval(state.timer); state.timer = null; }
      if (state.fade) { window.clearTimeout(state.fade); state.fade = null; }
      state.past = 0;
      state.phase = 'empty';
      state.busy = false;
      state.newPos = -1;
      render();
    }

    btnPre.addEventListener('click', prefill);
    btnDec.addEventListener('click', function () { if (!state.busy) { decodeStep(); } });
    btnAuto.addEventListener('click', autoDecode);
    btnReset.addEventListener('click', reset);

    // 预置：prefill 完成 + 第一步 decode 完成，保证打开即有内容。
    state.past = PROMPT.length;
    state.phase = 'prefilled';
    decodeStep();

    return function dispose() {
      if (state.timer) { window.clearInterval(state.timer); }
      if (state.fade) { window.clearTimeout(state.fade); }
      if (style.parentNode) { style.parentNode.removeChild(style); }
    };
  }

  window.DLWidget.register('kv-cache', mount);
})();
