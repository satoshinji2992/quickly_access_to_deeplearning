/* swiglu-gate — SwiGLU 三面板：SiLU 门曲线（对照 ReLU）、
 * 6 维向量的门控流水线（拖输入看每维开合）、参数量对齐计算器。
 */
(function () {
  'use strict';

  var INK = '#071321', BLUE = '#0b63f3', RED = '#c0392b', PAPER = '#fffef9', LIME = '#c8ff47';
  var MONO = 'ui-monospace,SFMono-Regular,Consolas,monospace';

  var sigmoid = function (z) { return 1 / (1 + Math.exp(-z)); };
  var silu = function (z) { return z * sigmoid(z); };

  /* 手选的 6 维权重：让门既有大开、也有接近关闭、还有小负值。 */
  var WG = [1.4, -1.1, 0.9, 0.4, -0.6, 1.2];
  var WU = [0.8, 1.2, -1.0, 1.5, 0.7, 0.5];
  var X0 = [1.6, 0.8, -0.6, 1.2, 0.4, -1.4];

  var CSS = '' +
    '.sg-curve{border:1px solid #d8d3c4;background:#fffef9}' +
    '.sg-grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:5px;margin:10px 0 4px}' +
    '.sg-cell{border:1px solid rgba(7,19,33,.3);background:#fffef9;padding:5px 4px;text-align:center}' +
    '.sg-cell i{display:block;font:700 8px ' + MONO + ';font-style:normal;color:#65717c}' +
    '.sg-cell b{display:block;font:800 11px ' + MONO + ';color:#071321;margin-top:2px;overflow:hidden;text-overflow:ellipsis}' +
    '.sg-rowlab{font:800 9px ' + MONO + ';color:#0b63f3;margin:10px 0 0;display:flex;justify-content:space-between}' +
    '.sg-sliders{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px;margin-top:6px}' +
    '.sg-sliders label{display:flex;flex-direction:column;gap:3px;font:700 8.5px ' + MONO + ';color:#65717c}' +
    '.sg-note{margin:8px 0 0}' +
    '.sg-two{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start;margin-top:4px}' +
    '@media(max-width:620px){.sg-two{grid-template-columns:1fr}.sg-grid,.sg-sliders{grid-template-columns:repeat(3,minmax(0,1fr))}}';

  function mount(container) {
    container.innerHTML =
      '<style>' + CSS + '</style>' +
      '<p class="wg-title">SwiGLU：一扇可以拧的门</p>' +
      '<p class="wg-sub">门 = SiLU(gate 分支)，内容 = up 分支，逐元素相乘后再投影回去。</p>' +

      '<div class="wg-block">' +
        '<div class="wg-label"><span>门曲线 · SiLU vs ReLU</span><span data-role="zlab"></span></div>' +
        '<canvas class="wg-canvas sg-curve" data-role="curve" height="190"></canvas>' +
        '<input class="wg-slider" type="range" data-role="z" min="-60" max="60" step="1" value="14">' +
        '<div class="wg-readout">' +
          '<div class="wg-stat"><span>z（门的输入）</span><b data-role="zv"></b></div>' +
          '<div class="wg-stat"><span>SiLU(z) 门开度</span><b data-role="sv"></b></div>' +
          '<div class="wg-stat"><span>ReLU(z) 只有开关</span><b data-role="rv"></b></div>' +
        '</div>' +
        '<p class="wg-note sg-note" data-role="cnote"></p>' +
      '</div>' +

      '<div class="wg-block">' +
        '<div class="wg-label"><span>6 维向量的门控流水线</span><span>x 可拖 · 权重固定</span></div>' +
        '<p class="wg-note">拖动输入 x 的各维，看门怎样逐维拧动：gate 接近 0 的维几乎关死，接近 1 的维原样放行，小负值则反着拧一点。</p>' +
        '<div class="wg-sliders" data-role="xsliders"></div>' +
        '<div class="sg-rowlab"><span>gate = SiLU(W_gate·x)</span><span>⊙</span></div>' +
        '<div class="sg-grid" data-role="grow"></div>' +
        '<div class="sg-rowlab"><span>up = W_up·x</span><span>=</span></div>' +
        '<div class="sg-grid" data-role="urow"></div>' +
        '<div class="sg-rowlab"><span>gate ⊙ up（进 down 投影前）</span><span data-role="pick"></span></div>' +
        '<div class="sg-grid" data-role="orow"></div>' +
      '</div>' +

      '<div class="wg-block">' +
        '<div class="wg-label"><span>参数量对齐：为什么取 8/3·d</span><span data-role="dlab"></span></div>' +
        '<p class="wg-note">普通 FFN 用 4d，参数 8d²；SwiGLU 三个矩阵，取 m≈8/3·d 才能持平。m 向上对齐到 256 的倍数（LLaMA 惯例）。</p>' +
        '<input class="wg-slider" type="range" data-role="d" min="512" max="8192" step="512" value="4096">' +
        '<div class="wg-readout">' +
          '<div class="wg-stat"><span>普通 FFN (m=4d)</span><b data-role="pffn"></b></div>' +
          '<div class="wg-stat"><span>SwiGLU m=8/3·d 对齐</span><b data-role="msg"></b></div>' +
          '<div class="wg-stat"><span>SwiGLU 参数 3dm</span><b data-role="psg"></b></div>' +
        '</div>' +
        '<p class="wg-note sg-note" data-role="pnote"></p>' +
      '</div>';

    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };

    /* ---- 面板 1：曲线 ---- */
    var cv = q('curve');
    var CURVE_H = 190;
    function dpr(canvas, logicalHeight) {
      var cssWidth = canvas.parentElement ? canvas.parentElement.clientWidth : 0;
      cssWidth = Math.max(220, Math.min(cssWidth || 400, 640));
      var ratio = window.devicePixelRatio || 1;
      canvas.style.width = cssWidth + 'px';
      canvas.style.height = logicalHeight + 'px';
      canvas.width = Math.round(cssWidth * ratio);
      canvas.height = Math.round(logicalHeight * ratio);
      var ctx = canvas.getContext('2d');
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return { ctx: ctx, w: cssWidth, h: logicalHeight };
    }
    function drawCurve() {
      var z = parseInt(q('z').value, 10) / 10;
      var c = dpr(cv, CURVE_H), ctx = c.ctx;
      var L = 30, R = 10, T = 10, B = 22;
      var pw = c.w - L - R, ph = c.h - T - B;
      var ZM = 6, YM = 6;
      var X = function (v) { return L + (v + ZM) / (2 * ZM) * pw; };
      var Y = function (v) { return T + (1 - (v + YM) / (2 * YM)) * ph; };
      ctx.clearRect(0, 0, c.w, c.h);
      ctx.strokeStyle = 'rgba(7,19,33,.12)'; ctx.lineWidth = 1;
      [0, 1, -1, 2, -2].forEach(function (gy) {
        ctx.beginPath(); ctx.moveTo(L, Y(gy)); ctx.lineTo(L + pw, Y(gy)); ctx.stroke();
      });
      ctx.strokeStyle = 'rgba(7,19,33,.35)';
      ctx.beginPath(); ctx.moveTo(L, Y(0)); ctx.lineTo(L + pw, Y(0)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(X(0), T); ctx.lineTo(X(0), T + ph); ctx.stroke();
      ctx.fillStyle = '#65717c'; ctx.font = '9px ' + MONO;
      ctx.textAlign = 'right'; ctx.fillText('+2', L - 3, Y(2) + 3); ctx.fillText('0', L - 3, Y(0) + 3); ctx.fillText('-2', L - 3, Y(-2) + 3);

      function plot(fn, style, dash, width) {
        ctx.strokeStyle = style; ctx.lineWidth = width || 2; ctx.setLineDash(dash || []);
        ctx.beginPath();
        for (var s = 0; s <= 120; s++) {
          var zz = -ZM + (s / 120) * 2 * ZM;
          var x = X(zz), y = Y(Math.max(-YM, Math.min(YM, fn(zz))));
          if (s) { ctx.lineTo(x, y); } else { ctx.moveTo(x, y); }
        }
        ctx.stroke(); ctx.setLineDash([]);
      }
      plot(function (v) { return Math.max(0, v); }, 'rgba(7,19,33,.4)', [5, 4], 1.6);
      plot(silu, BLUE, [], 2.4);

      var px = X(z);
      ctx.strokeStyle = 'rgba(11,99,243,.4)'; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(px, T); ctx.lineTo(px, T + ph); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = BLUE;
      ctx.beginPath(); ctx.arc(px, Y(silu(z)), 4.4, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = INK; ctx.lineWidth = 1.3; ctx.stroke();
      ctx.fillStyle = 'rgba(7,19,33,.5)';
      ctx.beginPath(); ctx.arc(px, Y(Math.max(0, z)), 3, 0, Math.PI * 2); ctx.fill();

      q('zlab').textContent = '拖滑杆移动蓝点';
      q('zv').textContent = z.toFixed(1);
      q('sv').textContent = silu(z).toFixed(3);
      q('rv').textContent = Math.max(0, z).toFixed(1);
      q('cnote').textContent = z < 0
        ? 'z=' + z.toFixed(1) + '：ReLU 已把这一维归零；SiLU 留了 ' + silu(z).toFixed(3) + ' 的小负值——门还有一点反向开度。'
        : 'z=' + z.toFixed(1) + '：门开到 ' + silu(z).toFixed(3) + '，内容按这个比例放行（1 = 原样通过）。';
    }

    /* ---- 面板 2：6 维流水线 ---- */
    var x = X0.slice();
    var xsliders = q('xsliders');
    X0.forEach(function (v0, i) {
      var lab = document.createElement('label');
      lab.innerHTML = 'x' + i + '<input class="wg-slider" type="range" min="-20" max="20" step="1" value="' +
        Math.round(v0 * 10) + '" data-i="' + i + '">';
      xsliders.appendChild(lab);
    });
    xsliders.addEventListener('input', function (e) {
      var t = e.target;
      if (t.dataset.i !== undefined) {
        x[parseInt(t.dataset.i, 10)] = parseInt(t.value, 10) / 10;
        paintVector();
      }
    });

    function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) { s += a[i] * b[i]; } return s; }
    function gateVec() { var g = []; for (var i = 0; i < 6; i++) { g.push(dot(x, WG.map(function (w, j) { return j === i ? w : 0; }))); } return g; }

    function cell(parent, label, value, colorFn) {
      var d = document.createElement('div');
      d.className = 'sg-cell';
      d.innerHTML = '<i>' + label + '</i><b>' + (value >= 0 ? '' : '−') + Math.abs(value).toFixed(2) + '</b>';
      var s = colorFn(value);
      d.querySelector('b').style.color = s.fg;
      d.style.background = s.bg;
      parent.appendChild(d);
    }
    function siluColor(v) {
      var t = Math.max(0, Math.min(1, Math.abs(v) / 2));
      if (v >= 0) { return { bg: 'rgba(11,99,243,' + (0.10 + 0.5 * Math.min(1, v)) + ')', fg: INK }; }
      return { bg: 'rgba(192,57,43,' + (0.10 + 0.4 * t) + ')', fg: INK };
    }
    function plainColor(v) {
      var t = Math.max(0, Math.min(1, Math.abs(v) / 2));
      var base = v >= 0 ? '11,99,243' : '192,57,43';
      return { bg: 'rgba(' + base + ',' + (0.08 + 0.45 * t).toFixed(2) + ')', fg: INK };
    }

    function paintVector() {
      /* g_i = WG[i]·x 的直觉化写法：每维只受 x 同维影响（对角简化），便于看开合。 */
      var grow = q('grow'), urow = q('urow'), orow = q('orow');
      grow.innerHTML = ''; urow.innerHTML = ''; orow.innerHTML = '';
      var gates = [], ups = [], outs = [];
      for (var i = 0; i < 6; i++) {
        var gPre = WG[i] * x[i];
        var g = silu(gPre);
        var u = WU[i] * x[i];
        gates.push(g); ups.push(u); outs.push(g * u);
      }
      for (i = 0; i < 6; i++) { cell(grow, 'h' + i, gates[i], siluColor); }
      for (i = 0; i < 6; i++) { cell(urow, 'h' + i, ups[i], plainColor); }
      for (i = 0; i < 6; i++) { cell(orow, 'h' + i, outs[i], plainColor); }
      var open = 0;
      gates.forEach(function (g) { open += Math.abs(g) > 0.5 ? 1 : 0; });
      q('pick').textContent = '门大开 ' + open + '/6 维';
    }

    /* ---- 面板 3：参数量 ---- */
    function fmtM(n) {
      return n >= 1e9 ? (n / 1e9).toFixed(2) + 'B' : (n / 1e6).toFixed(1) + 'M';
    }
    function paintParams() {
      var d = parseInt(q('d').value, 10);
      var mIdeal = d * 8 / 3;
      var m = Math.ceil(mIdeal / 256) * 256;
      var pFfn = 8 * d * d;
      var pSg = 3 * d * m;
      q('dlab').textContent = 'd = ' + d;
      q('pffn').textContent = fmtM(pFfn);
      q('msg').textContent = m + '（理想 ' + Math.round(mIdeal) + '）';
      q('psg').textContent = fmtM(pSg);
      q('pnote').textContent = 'd=' + d + '：普通 FFN ' + fmtM(pFfn) + '，SwiGLU ' + fmtM(pSg) +
        '（差 ' + ((pSg / pFfn - 1) * 100).toFixed(1) + '%）' +
        (d === 4096 ? '——正是 LLaMA 的取值：理想 10923，对齐到 11008。' : '。');
    }

    q('z').addEventListener('input', drawCurve);
    q('d').addEventListener('input', paintParams);
    var onResize = function () { drawCurve(); };
    window.addEventListener('resize', onResize);

    drawCurve();
    paintVector();
    paintParams();
    window.setTimeout(function () { drawCurve(); }, 60);

    return function dispose() {
      window.removeEventListener('resize', onResize);
    };
  }

  window.DLWidget.register('swiglu-gate', mount);
})();
