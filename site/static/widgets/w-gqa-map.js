/* gqa-map — GQA 分组映射：4 个 query head 共享 2 组 KV，repeat_kv 展开与缓存体积对比。
 *
 * 设定：Hq=4、Hkv=2、group=2。q0、q1 → 组 0（K0·V0），q2、q3 → 组 1（K1·V1）。
 * 「处理一个 token」动画：token 进入 → 4 个 q 同时激活 → KV 复制展开（虚线副本）
 * → 输出 o0..o3。下方对比 MHA / GQA 的 KV cache 体积（比例 = Hkv/Hq）。
 */
(function () {
  'use strict';

  var TOKENS = ['猫', '在', '睡', '觉'];
  var MONO = 'ui-monospace,SFMono-Regular,Consolas,monospace';
  var KV0 = '#0a3f9e';   // 组 0 深蓝
  var KV1 = '#72d7ff';   // 组 1 浅蓝
  var LANE_Y = [78, 124, 170, 216];

  var CSS = '' +
    '.gq-legend{display:flex;flex-wrap:wrap;gap:8px 14px;margin:2px 0 10px;font:700 9px ' + MONO + ';color:#65717c;align-items:center}' +
    '.gq-lch{display:inline-block;width:13px;height:10px;border:1px solid rgba(7,19,33,.4);margin-right:4px;vertical-align:-1px}' +
    '.gq-lch.dash{border-style:dashed;background:rgba(10,63,158,.12)}' +
    '.gq-diagram{border:1px solid #d8d3c4;background:#f7f5ef;padding:4px 2px 0}' +
    '.gq-diagram svg{display:block;width:100%;height:auto}' +
    '.gq-head{font:700 9px ' + MONO + ';fill:#65717c;letter-spacing:.05em;text-anchor:middle}' +
    '.gq-head2{font:700 8px ' + MONO + ';fill:#65717c;text-anchor:middle}' +
    '.gq-chip{fill:#fffef9;stroke:#071321;stroke-width:1.5}' +
    '.gq-toktxt{font:800 13px ' + MONO + ';fill:#071321;text-anchor:middle}' +
    '.gq-toksub{font:700 8px ' + MONO + ';fill:#65717c;text-anchor:middle}' +
    '.gq-fan{stroke:rgba(7,19,33,.5);stroke-width:1;opacity:0;transition:opacity .35s ease}' +
    '.s-enter .gq-fan{opacity:.55}' +
    '.gq-tok{opacity:0;transform:translateX(-72px);transition:opacity .45s cubic-bezier(.2,.7,.3,1),transform .45s cubic-bezier(.2,.7,.3,1)}' +
    '.s-enter .gq-tok{opacity:1;transform:none}' +
    '.gq-halo{fill:#0b63f3;opacity:0;transition:opacity .3s ease}' +
    '.s-q .gq-halo{opacity:.22}' +
    '.gq-qrect{fill:#0b63f3;stroke:#071321;stroke-width:1.5;transition:stroke-width .25s ease}' +
    '.s-q .gq-qrect{stroke-width:2.5}' +
    '.gq-qtxt{font:800 12px ' + MONO + ';fill:#fffef9;text-anchor:middle}' +
    '.gq-line{stroke:rgba(7,19,33,.28);stroke-width:1.5;transition:stroke .3s ease,stroke-width .3s ease}' +
    '.s-q .gq-line{stroke:#0b63f3;stroke-width:2}' +
    '.gq-larr{fill:rgba(7,19,33,.3);transition:fill .3s ease}' +
    '.s-q .gq-larr{fill:#0b63f3}' +
    '.gq-rpt{opacity:0;transform:scale(.72);transform-box:fill-box;transform-origin:center;transition:opacity .3s ease,transform .3s ease}' +
    '.gq-rpt.gq-d1{transition-delay:.18s}' +
    '.s-rpt .gq-rpt{opacity:1;transform:none}' +
    '.gq-rrect{fill-opacity:.16;stroke:#071321;stroke-width:1;stroke-dasharray:4 3}' +
    '.gq-rtxt{font:800 9.5px ' + MONO + ';text-anchor:middle}' +
    '.gq-kvblock{fill:#fffef9;stroke:#071321;stroke-width:1.5}' +
    '.s-rpt .gq-kvblock{animation:gq-pulse .55s ease}' +
    '@keyframes gq-pulse{0%{stroke-width:1.5}35%{stroke-width:4}100%{stroke-width:1.5}}' +
    '.gq-blabel{font:700 8.5px ' + MONO + ';fill:#65717c;text-anchor:middle}' +
    '.gq-kvtxt{font:800 9.5px ' + MONO + ';text-anchor:middle}' +
    '.gq-oarr{opacity:0;transition:opacity .35s ease}' +
    '.s-out .gq-oarr{opacity:1}' +
    '.gq-oaline{stroke:#0b63f3;stroke-width:2}' +
    '.gq-oapoly{fill:#0b63f3}' +
    '.gq-ochip{opacity:0;transform:translateX(14px);transform-box:fill-box;transition:opacity .35s ease,transform .35s ease}' +
    '.s-out .gq-ochip{opacity:1;transform:none}' +
    '.gq-orect{fill:#c8ff47;stroke:#071321;stroke-width:1.5;transition:fill .5s ease}' +
    '.gq-otxt{font:800 10.5px ' + MONO + ';fill:#071321;text-anchor:middle;transition:fill .5s ease}' +
    '.s-done .gq-orect{fill:#fffef9}' +
    '.s-done .gq-otxt{fill:#0b63f3}' +
    '.gq-shape{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:12px 0 2px;padding:8px 10px;border:1px dashed rgba(7,19,33,.35);background:#fffef9;font:700 10.5px ' + MONO + ';color:#45515c}' +
    '.gq-schip{border:1px solid #071321;background:#fffef9;padding:2px 6px;color:#071321;white-space:nowrap}' +
    '.gq-schip.gq-sq{background:#0b63f3;color:#fffef9}' +
    '.gq-schip.gq-so{color:#0b63f3;border-color:#0b63f3}' +
    '.gq-op{color:#65717c}' +
    '.gq-sgroup{color:#65717c;margin-left:auto}' +
    '.gq-bars{border:1px solid rgba(7,19,33,.25);background:#f4f2ec;padding:10px;display:flex;flex-direction:column;gap:8px}' +
    '.gq-brow{display:flex;align-items:center;gap:8px}' +
    '.gq-bname{flex:0 0 104px;font:700 9.5px ' + MONO + ';color:#45515c;text-align:right}' +
    '.gq-btrack{flex:1 1 auto;display:flex;align-items:center;gap:6px;min-width:0}' +
    '.gq-bar{height:24px;display:flex;align-items:center;justify-content:center;border:1px solid #071321;overflow:hidden}' +
    '.gq-bar b{font:800 9.5px ' + MONO + ';color:#fffef9;white-space:nowrap}' +
    '.gq-bmha{background:#263541}' +
    '.gq-bgqa{background:#0b63f3}' +
    '.gq-save{font:800 9px ' + MONO + ';color:#071321;background:#c8ff47;border:1px solid #071321;padding:3px 6px;white-space:nowrap}' +
    '.gq-ctrl .wg-button:disabled{opacity:.4;cursor:default}' +
    '.gq-ctrl .wg-button:disabled:hover{background:#fffef9;color:#071321;box-shadow:none}' +
    '.gq-ctrl .wg-button.is-primary:disabled:hover{background:#071321;color:#c8ff47}';

  function headers() {
    return '' +
      '<text class="gq-head" x="35" y="14">TOKEN</text>' +
      '<text class="gq-head" x="113" y="14">Q ×4</text><text class="gq-head2" x="113" y="25">Hq = 4</text>' +
      '<text class="gq-head" x="340" y="14">REPEAT_KV</text><text class="gq-head2" x="340" y="25">复制展开 ×2</text>' +
      '<text class="gq-head" x="527" y="14">KV ×2 共享</text><text class="gq-head2" x="527" y="25">Hkv = 2</text>' +
      '<text class="gq-head" x="623" y="14">OUT ×4</text>';
  }

  function tokenGroup() {
    var fan = '';
    for (var i = 0; i < 4; i += 1) {
      fan += '<line class="gq-fan" x1="64" y1="147" x2="96" y2="' + LANE_Y[i] + '"/>';
    }
    return '<g class="gq-tok">' +
      '<rect class="gq-chip" x="6" y="134" width="58" height="26" rx="3"/>' +
      '<text class="gq-toktxt" x="35" y="151">猫</text>' +
      '<text class="gq-toksub" x="35" y="171">x_t (B,1,D)</text>' +
      '</g>' + fan;
  }

  function lanes() {
    var s = '';
    for (var i = 0; i < 4; i += 1) {
      var y = LANE_Y[i], g = i < 2 ? 0 : 1;
      var gc = g === 0 ? KV0 : KV1, tc = g === 0 ? KV0 : '#071321';
      s += '' +
        '<rect class="gq-halo" x="92" y="' + (y - 21) + '" width="42" height="42" rx="6"/>' +
        '<line class="gq-line" x1="132" y1="' + y + '" x2="456" y2="' + y + '"/>' +
        '<polygon class="gq-larr" points="455,' + (y - 4.5) + ' 468,' + y + ' 455,' + (y + 4.5) + '"/>' +
        '<rect class="gq-qrect" x="96" y="' + (y - 17) + '" width="34" height="34" rx="4"/>' +
        '<text class="gq-qtxt" x="113" y="' + (y + 4) + '">q' + i + '</text>' +
        '<g class="gq-rpt' + (g === 1 ? ' gq-d1' : '') + '">' +
          '<rect class="gq-rrect" x="306" y="' + (y - 13) + '" width="68" height="26" fill="' + gc + '"/>' +
          '<text class="gq-rtxt" x="340" y="' + (y + 3.5) + '" fill="' + tc + '">K' + g + '·V' + g + '</text>' +
        '</g>' +
        '<g class="gq-oarr">' +
          '<line class="gq-oaline" x1="584" y1="' + y + '" x2="590" y2="' + y + '"/>' +
          '<polygon class="gq-oapoly" points="588,' + (y - 4) + ' 598,' + y + ' 588,' + (y + 4) + '"/>' +
        '</g>' +
        '<g class="gq-ochip">' +
          '<rect class="gq-orect" x="600" y="' + (y - 12) + '" width="46" height="24" rx="3"/>' +
          '<text class="gq-otxt" x="623" y="' + (y + 4) + '">o' + i + '</text>' +
        '</g>';
    }
    return s;
  }

  function kvBlock(g) {
    var y0 = g === 0 ? 64 : 156;
    var fill = g === 0 ? KV0 : KV1;
    var tc = g === 0 ? '#fffef9' : '#071321';
    return '<g>' +
      '<rect class="gq-kvblock" x="470" y="' + y0 + '" width="114" height="74"/>' +
      '<text class="gq-blabel" x="527" y="' + (y0 + 15) + '">组 ' + g + ' · group=2</text>' +
      '<rect x="484" y="' + (y0 + 24) + '" width="27" height="27" rx="3" fill="' + fill + '"/>' +
      '<text class="gq-kvtxt" x="497.5" y="' + (y0 + 41) + '" fill="' + tc + '">K' + g + '</text>' +
      '<rect x="543" y="' + (y0 + 24) + '" width="27" height="27" rx="3" fill="' + fill + '"/>' +
      '<text class="gq-kvtxt" x="556.5" y="' + (y0 + 41) + '" fill="' + tc + '">V' + g + '</text>' +
      '<text class="gq-blabel" x="527" y="' + (y0 + 66) + '">K' + g + '/V' + g + ' 各 (T,Dh)</text>' +
      '</g>';
  }

  function mount(container) {
    container.innerHTML =
      '<style>' + CSS + '</style>' +
      '<p class="wg-title">GQA：多个 query head 共享一组 KV</p>' +
      '<p class="wg-sub">q0、q1 共享组 0，q2、q3 共享组 1</p>' +
      '<div class="gq-legend">' +
        '<span><i class="gq-lch" style="background:#0b63f3"></i>query heads ×4</span>' +
        '<span><i class="gq-lch" style="background:' + KV0 + '"></i>组 0 K·V</span>' +
        '<span><i class="gq-lch" style="background:' + KV1 + '"></i>组 1 K·V</span>' +
        '<span><i class="gq-lch dash"></i>repeat_kv 副本</span>' +
        '<span><i class="gq-lch" style="background:#c8ff47"></i>本步新出现</span>' +
      '</div>' +
      '<div class="gq-diagram" data-role="diag">' +
        '<svg viewBox="0 0 660 252" role="img" aria-label="GQA 分组映射：q0 q1 共享组 0，q2 q3 共享组 1">' +
          headers() + tokenGroup() + lanes() + kvBlock(0) + kvBlock(1) +
        '</svg>' +
      '</div>' +
      '<div class="gq-shape">' +
        '<span class="gq-schip gq-sq">q (B,4,T,Dh)</span>' +
        '<span class="gq-op">＋</span>' +
        '<span class="gq-schip">K,V (B,2,T,Dh)</span>' +
        '<span class="gq-op">─ repeat_kv ─▶ (B,4,T,Dh)</span>' +
        '<span class="gq-op">＝</span>' +
        '<span class="gq-schip gq-so">o (B,4,T,Dh)</span>' +
        '<span class="gq-sgroup">group = Hq/Hkv = 2</span>' +
      '</div>' +
      '<div class="wg-readout">' +
        '<div class="wg-stat"><span>query heads · Hq</span><b>4</b></div>' +
        '<div class="wg-stat"><span>kv heads · Hkv</span><b>2</b></div>' +
        '<div class="wg-stat"><span>group = Hq/Hkv</span><b>2</b></div>' +
        '<div class="wg-stat"><span>KV cache vs MHA</span><b>1/2</b></div>' +
      '</div>' +
      '<div class="wg-label"><span>KV cache 体积对比</span><span>head_dim 相同 · 单层 · T 个 token</span></div>' +
      '<div class="gq-bars">' +
        '<div class="gq-brow"><span class="gq-bname">MHA · 4 组 KV</span>' +
          '<div class="gq-btrack"><div class="gq-bar gq-bmha" style="width:100%"><b>2×4×T×Dh = 8·T·Dh</b></div></div></div>' +
        '<div class="gq-brow"><span class="gq-bname">GQA · 2 组 KV</span>' +
          '<div class="gq-btrack"><div class="gq-bar gq-bgqa" style="width:50%"><b>2×2×T×Dh = 4·T·Dh</b></div><span class="gq-save">省 50%</span></div></div>' +
      '</div>' +
      '<p class="wg-note">缓存比例 = Hkv/Hq，与 T、Dh 无关</p>' +
      '<div class="wg-controls gq-ctrl">' +
        '<button type="button" class="wg-button is-primary" data-role="step">处理一个 token</button>' +
        '<button type="button" class="wg-button" data-role="reset">重置</button>' +
      '</div>' +
      '<p class="wg-note" data-role="note"></p>';

    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };
    var diag = q('diag'), noteEl = q('note'), tokTxt = container.querySelector('.gq-toktxt');
    var btnStep = q('step'), btnReset = q('reset');
    var STAGES = ['enter', 'q', 'rpt', 'out', 'done'];
    var state = { count: 1, busy: false, timers: [] };

    function after(ms, fn) { state.timers.push(window.setTimeout(fn, ms)); }
    function clearTimers() {
      state.timers.forEach(function (id) { window.clearTimeout(id); });
      state.timers = [];
    }
    function buttons() {
      btnStep.disabled = state.busy;
      btnReset.disabled = state.busy;
    }
    function setStage(n) {
      var cls = 'gq-diagram';
      for (var i = 0; i <= n; i += 1) { cls += ' s-' + STAGES[i]; }
      diag.className = cls;
    }
    function note(text) { noteEl.textContent = text; }

    function runToken() {
      if (state.busy) { return; }
      state.busy = true;
      buttons();
      var tok = TOKENS[state.count % TOKENS.length];
      tokTxt.textContent = tok;
      setStage(-1);
      after(240, function () {
        setStage(0);
        note('token 进入：Wq 一次投影出 4 份 query');
      });
      after(700, function () {
        setStage(1);
        note('4 个 q 同时激活，各自独立');
      });
      after(1280, function () {
        setStage(2);
        note('repeat_kv 复制 KV，存储仍 2 组');
      });
      after(2120, function () {
        setStage(3);
        note('各组 q 对本组 K/V 做注意力，得 o0..o3');
      });
      after(3220, function () {
        setStage(4);
        state.count += 1;
        state.busy = false;
        buttons();
        note('已处理 ' + state.count + ' 个，缓存仍 2 组；下一个「' + TOKENS[state.count % TOKENS.length] + '」');
      });
    }

    function reset() {
      clearTimers();
      state.busy = false;
      state.count = 1;
      tokTxt.textContent = TOKENS[0];
      buttons();
      setStage(-1);
      after(280, function () {
        setStage(4);
        note('点「处理一个 token」重放动画');
      });
    }

    btnStep.addEventListener('click', runToken);
    btnReset.addEventListener('click', reset);

    // 预置到"已处理 1 个 token"的完成态，保证打开即有内容。
    setStage(4);
    note('点「处理一个 token」重放动画');

    return function dispose() { clearTimers(); };
  }

  window.DLWidget.register('gqa-map', mount);
})();
