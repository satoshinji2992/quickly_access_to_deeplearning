/* rope-rotate — RoPE 旋转位置编码：位置 m 不进向量，而是变成 Q/K 相邻两维的旋转角 m·θ。
 * 单位圆上把同一初始向量旋转到不同位置，点积只依赖相对距离 Δ=n−m。D=16、base=10000。
 */
(function () {
  'use strict';

  var INK = '#071321', BLUE = '#0b63f3', PAPER = '#fffef9', GREEN = '#c8ff47';
  var MONO = 'ui-monospace,SFMono-Regular,Consolas,monospace';
  var D = 16, NP = 8, MAXT = 8, TAU = Math.PI * 2;

  function omega(i) { return Math.pow(10000, -2 * i / D); }          // θ_i，i = 0..7

  function fmtW(w) { return w >= 0.01 ? w.toFixed(2) : w.toExponential(1); }
  function fmtDeg(rad) { var d = rad * 180 / Math.PI; return (d >= 0.1 ? d.toFixed(1) : d.toFixed(2)) + '°'; }
  function fmtCos(v) {   // θ 很小时 cos ≈ 1，直接标 ≈1 更诚实；负号统一用排版减号
    if (v > 0.9999) { return '≈1'; }
    if (v < -0.9999) { return '≈−1'; }
    return (Math.abs(v) >= 0.999 ? v.toFixed(4) : v.toFixed(3)).replace('-', '−');
  }
  function showCh(ch) { return /^\s$/.test(ch) ? '␣' : ch; }

  var CSS = '' +
    '.rr-row{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start;margin:12px 0}' +
    '.rr-in{width:100%;box-sizing:border-box;border:1px solid #071321;background:#fffef9;padding:8px 10px;font:600 14px/1.5 ' + MONO + ';color:#071321}' +
    '.rr-in:focus{outline:none;box-shadow:3px 3px 0 rgba(11,99,243,.55)}' +
    '.rr-col-cir{flex:1 1 270px;min-width:250px}' +
    '.rr-col-ctl{flex:1 1 230px;min-width:225px}' +
    '.rr-leg{display:flex;flex-wrap:wrap;gap:8px 14px;margin:9px 0 0;font:700 9px ' + MONO + ';color:#65717c;align-items:center}' +
    '.rr-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:-1px}' +
    '.rr-dot.g{background:#c8ff47;border:1.5px solid #071321}' +
    '.rr-dot.b{background:#0b63f3}' +
    '.rr-arc{display:inline-block;width:14px;height:9px;border:2px solid #0b63f3;border-radius:50% 50% 0 0/100% 100% 0 0;border-bottom:none;margin-right:4px;vertical-align:-1px}' +
    '.rr-dt{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:4px;margin:4px 0 2px}' +
    '.rr-dtc{border:1px solid rgba(7,19,33,.25);background:#fffef9;padding:4px 2px;text-align:center;line-height:1.35}' +
    '.rr-dtc i{display:block;font:700 8px ' + MONO + ';font-style:normal;color:#65717c}' +
    '.rr-dtc b{font:800 11.5px ' + MONO + ';color:#071321}';

  function mount(container) {
    container.innerHTML =
      '<style>' + CSS + '</style>' +
      '<p class="wg-title">RoPE：位置变成旋转角</p>' +
      '<p class="wg-sub">正弦编码把位置向量加到 embedding 上；RoPE 不动 embedding，而是把 Q、K 的相邻两维看成平面向量，按位置旋转 m·θ。取 D=16（8 个维度对）、base=10000。</p>' +
      '<div class="wg-label"><span>句子（按字符切分，最多 8 字）</span><span data-role="cnt"></span></div>' +
      '<input class="rr-in" data-role="sent" value="小猫在睡觉">' +
      '<p class="wg-note" data-role="snote"></p>' +
      '<div class="rr-row">' +
        '<div class="rr-col-cir">' +
          '<div class="wg-label"><span>单位圆 · 位置 m 的向量 = 初始向量旋转 m·θ</span><span data-role="tott"></span></div>' +
          '<canvas class="wg-canvas" data-role="cir" height="316"></canvas>' +
          '<div class="rr-leg">' +
            '<span><i class="rr-dot g"></i>m=0 基准（不旋转）</span>' +
            '<span><i class="rr-dot b"></i>m=1..T−1</span>' +
            '<span><i class="rr-arc"></i>每步转角 θ</span>' +
          '</div>' +
        '</div>' +
        '<div class="rr-col-ctl">' +
          '<div class="wg-label"><span>维度对 i（θ 随 i 变小）</span><span data-role="thlab"></span></div>' +
          '<input class="wg-slider" type="range" data-role="pair" min="0" max="7" step="1" value="0">' +
          '<div class="wg-readout">' +
            '<div class="wg-stat"><span>θi（rad / 度）</span><b data-role="thv"></b></div>' +
            '<div class="wg-stat"><span>相邻点积 cos θ</span><b data-role="cth"></b></div>' +
          '</div>' +
          '<div class="wg-label"><span>任意两点积 = cos(Δ·θ)</span><span>Δ = n − m</span></div>' +
          '<div class="rr-dt" data-role="dt"></div>' +
          '<div class="wg-readout" data-role="eqrow">' +
            '<div class="wg-stat"><span data-role="e1l"></span><b data-role="e1"></b></div>' +
            '<div class="wg-stat"><span data-role="e2l"></span><b data-role="e2"></b></div>' +
          '</div>' +
          '<p class="wg-note">同一初始向量旋转到位置 m 和 n，点积 = cos((n−m)·θ)，只依赖相对距离 Δ——这就是相对位置编码。换句子改变 T，这些读数一个都不会变。</p>' +
        '</div>' +
      '</div>' +
      '<p class="wg-note">把每个维度对 (x,y) 看成平面向量：位置 m 就是把它旋转 m·θi。θi = 10000^(−2i/D)，i 小转得快、i 大转得慢。旋转不改变长度和夹角，所以注意力打分里的 q·k 只看两 token 的相对距离，与绝对位置无关。V 不旋转——位置只影响「query 与哪个 key 匹配」，不影响取回的内容。</p>';

    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };
    var cvCir = q('cir');
    var state = { pair: 0, tokens: [] };
    var timers = [];
    function later(fn, ms) { timers.push(window.setTimeout(fn, ms)); }

    // HiDPI：显式钉住 CSS 尺寸，画布按 devicePixelRatio 放大（逻辑尺寸恒为常量）。
    function dpr(canvas, logicalHeight) {
      var cssWidth = canvas.parentElement ? canvas.parentElement.clientWidth : 0;
      cssWidth = Math.max(200, Math.min(cssWidth || 400, 640));
      var ratio = window.devicePixelRatio || 1;
      canvas.style.width = cssWidth + 'px';
      canvas.style.height = logicalHeight + 'px';
      canvas.width = Math.round(cssWidth * ratio);
      canvas.height = Math.round(logicalHeight * ratio);
      var ctx = canvas.getContext('2d');
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return { ctx: ctx, w: cssWidth, h: logicalHeight };
    }

    function halo(ctx, text, x, y, fill, font) {
      ctx.font = font; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.lineJoin = 'round'; ctx.lineWidth = 3.5; ctx.strokeStyle = 'rgba(255,254,249,.92)';
      ctx.strokeText(text, x, y);
      ctx.fillStyle = fill; ctx.fillText(text, x, y);
    }

    function el(tag, cls, text) {
      var node = document.createElement(tag);
      if (cls) { node.className = cls; }
      if (text !== undefined) { node.textContent = text; }
      return node;
    }

    function tokenize() {
      var raw = q('sent').value;
      var all = Array.from(raw);
      var note;
      if (!all.length) {
        state.tokens = ['□'];
        note = '输入为空：暂用占位 token □ 演示（T=1）。';
      } else {
        state.tokens = all.slice(0, MAXT);
        note = all.length > MAXT
          ? '句子过长：切分出 ' + all.length + ' 个字符，已截断为前 ' + MAXT + ' 个。'
          : 'T=' + state.tokens.length + '：每个字符一个位置 m=0..' + (state.tokens.length - 1) + '。';
      }
      q('cnt').textContent = state.tokens.length + ' / ' + MAXT + ' 字符';
      q('snote').textContent = note;
    }

    function drawCircle() {
      var T = state.tokens.length, th = omega(state.pair);
      var H = 316;
      var c = dpr(cvCir, H), ctx = c.ctx;
      var cx = c.w / 2, cy = c.h / 2 + 4;
      var R = Math.min(c.w, c.h) / 2 - 40;
      ctx.clearRect(0, 0, c.w, c.h);

      ctx.strokeStyle = 'rgba(7,19,33,.15)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(6, cy); ctx.lineTo(c.w - 6, cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx, 6); ctx.lineTo(cx, c.h - 6); ctx.stroke();
      ctx.strokeStyle = 'rgba(7,19,33,.8)'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.stroke();

      if (T >= 2 && th > 0.05) {
        var r2 = R * 0.44;
        ctx.strokeStyle = BLUE; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(cx, cy, r2, 0, -th, true); ctx.stroke();
        var mid = th / 2;
        halo(ctx, 'θ', cx + Math.cos(mid) * (r2 + 11), cy - Math.sin(mid) * (r2 + 11), BLUE, 'italic 800 12px ' + MONO);
      }

      var dirs = [];
      for (var m = 0; m < T; m++) {
        var a = m * th;
        dirs.push({ m: m, a: a, x: cx + Math.cos(a) * R, y: cy - Math.sin(a) * R });
      }
      dirs.forEach(function (p) {
        ctx.strokeStyle = p.m === 0 ? INK : 'rgba(7,19,33,.5)';
        ctx.lineWidth = p.m === 0 ? 2.2 : 1.4;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(p.x, p.y); ctx.stroke();
        var dx = p.x - cx, dy = p.y - cy, len = Math.sqrt(dx * dx + dy * dy) || 1;
        var ux = dx / len, uy = dy / len;
        ctx.fillStyle = p.m === 0 ? INK : 'rgba(7,19,33,.5)';
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x - ux * 8 - uy * 3.5, p.y - uy * 8 + ux * 3.5);
        ctx.lineTo(p.x - ux * 8 + uy * 3.5, p.y - uy * 8 - ux * 3.5);
        ctx.closePath(); ctx.fill();
      });
      dirs.forEach(function (p) {
        ctx.beginPath(); ctx.arc(p.x, p.y, p.m === 0 ? 4.2 : 3.2, 0, TAU);
        ctx.fillStyle = p.m === 0 ? GREEN : BLUE; ctx.fill();
        ctx.strokeStyle = INK; ctx.lineWidth = 1.3; ctx.stroke();
      });
      var lastLa = null;
      dirs.forEach(function (p) {
        var la = p.a, rr = R + 13;
        if (lastLa !== null && la - lastLa < 0.3) {
          if (p.m !== T - 1) { return; }
          rr = R + 27;
        }
        halo(ctx, String(p.m), cx + Math.cos(la) * rr, cy - Math.sin(la) * rr, INK, '800 10px ' + MONO);
        lastLa = la;
      });
    }

    function render() {
      var T = state.tokens.length, th = omega(state.pair);
      q('thlab').textContent = 'i=' + state.pair + ' · θ=' + fmtW(th);
      q('pair').value = String(state.pair);
      q('thv').textContent = fmtW(th) + ' / ' + fmtDeg(th);
      q('cth').textContent = fmtCos(Math.cos(th));
      q('tott').textContent = T >= 2 ? '总旋转 (T−1)·θ = ' + ((T - 1) * th).toFixed(2) + ' rad' : 'T=1';
      var dt = q('dt');
      dt.innerHTML = '';
      var span = Math.min(4, T - 1);
      if (span >= 1) {
        for (var dd = 1; dd <= span; dd++) {
          var cell = el('div', 'rr-dtc');
          cell.appendChild(el('i', '', 'Δ=' + dd));
          cell.appendChild(el('b', '', fmtCos(Math.cos(dd * th))));
          dt.appendChild(cell);
        }
      } else {
        dt.appendChild(el('div', 'wg-note', 'T=1：没有第二个 token，暂时没有点积可看。'));
      }
      if (T >= 4) {
        q('eqrow').style.display = '';
        q('e1l').textContent = 'q0·q3（Δ=3）';
        q('e2l').textContent = 'q' + (T - 4) + '·q' + (T - 1) + '（Δ=3）';
        q('e1').textContent = fmtCos(Math.cos(3 * th));
        q('e2').textContent = fmtCos(Math.cos(3 * th));
      } else {
        q('eqrow').style.display = 'none';
      }
      drawCircle();
    }

    q('sent').addEventListener('input', function () { tokenize(); render(); });
    q('pair').addEventListener('input', function () { state.pair = parseInt(this.value, 10); render(); });

    var onResize = function () { render(); };
    window.addEventListener('resize', onResize);

    tokenize();
    render();
    later(function () { render(); }, 60);   // 字体/布局就绪后再画一帧，防首轮宽度偏差

    return function dispose() {
      timers.forEach(function (t) { window.clearTimeout(t); });
      timers = [];
      window.removeEventListener('resize', onResize);
    };
  }

  window.DLWidget.register('rope-rotate', mount);
})();
