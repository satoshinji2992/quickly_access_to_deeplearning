/* swiglu-gate — SwiGLU 的门：SiLU 曲线对照 ReLU，拖 z 看门的开度。
 * 参数量推导见正文；这里只讲"门"这一个概念。
 */
(function () {
  'use strict';

  var INK = '#071321', BLUE = '#0b63f3', PAPER = '#fffef9';
  var MONO = 'ui-monospace,SFMono-Regular,Consolas,monospace';
  var CURVE_H = 220;

  var sigmoid = function (z) { return 1 / (1 + Math.exp(-z)); };
  var silu = function (z) { return z * sigmoid(z); };

  function mount(container) {
    container.innerHTML =
      '<p class="wg-title">SwiGLU 的门：SiLU</p>' +
      '<p class="wg-sub">gate 分支经过 SiLU 再乘到内容上。拖动 z，看门开到多少。</p>' +
      '<canvas class="wg-canvas" data-role="curve" height="220" style="border:1px solid #d8d3c4;background:#fffef9"></canvas>' +
      '<input class="wg-slider" type="range" data-role="z" min="-60" max="60" step="1" value="14" aria-label="门的输入 z">' +
      '<div class="wg-readout">' +
        '<div class="wg-stat"><span>z（门输入）</span><b data-role="zv"></b></div>' +
        '<div class="wg-stat"><span>SiLU(z) 开度</span><b data-role="sv"></b></div>' +
        '<div class="wg-stat"><span>ReLU(z) 开关</span><b data-role="rv"></b></div>' +
      '</div>' +
      '<p class="wg-note" data-role="note"></p>';

    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };
    var cv = q('curve');

    function draw() {
      var z = parseInt(q('z').value, 10) / 10;
      var cssWidth = cv.parentElement ? cv.parentElement.clientWidth : 0;
      cssWidth = Math.max(240, Math.min(cssWidth || 420, 640));
      var ratio = window.devicePixelRatio || 1;
      cv.style.width = cssWidth + 'px';
      cv.style.height = CURVE_H + 'px';
      cv.width = Math.round(cssWidth * ratio);
      cv.height = Math.round(CURVE_H * ratio);
      var ctx = cv.getContext('2d');
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      var w = cssWidth, h = CURVE_H;
      var L = 34, R = 10, T = 12, B = 24;
      var pw = w - L - R, ph = h - T - B, ZM = 6, YM = 6;
      var X = function (v) { return L + (v + ZM) / (2 * ZM) * pw; };
      var Y = function (v) { return T + (1 - (v + YM) / (2 * YM)) * ph; };
      ctx.clearRect(0, 0, w, h);

      ctx.strokeStyle = 'rgba(7,19,33,.10)'; ctx.lineWidth = 1;
      [2, 1, -1, -2].forEach(function (g) {
        ctx.beginPath(); ctx.moveTo(L, Y(g)); ctx.lineTo(L + pw, Y(g)); ctx.stroke();
      });
      ctx.strokeStyle = 'rgba(7,19,33,.35)';
      ctx.beginPath(); ctx.moveTo(L, Y(0)); ctx.lineTo(L + pw, Y(0)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(X(0), T); ctx.lineTo(X(0), T + ph); ctx.stroke();
      ctx.fillStyle = '#65717c'; ctx.font = '9px ' + MONO; ctx.textAlign = 'right';
      [2, 0, -2].forEach(function (g) { ctx.fillText(String(g).replace('-', '−'), L - 4, Y(g) + 3); });

      function plot(fn, style, dash, width) {
        ctx.strokeStyle = style; ctx.lineWidth = width; ctx.setLineDash(dash || []);
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
      ctx.beginPath(); ctx.arc(px, Y(silu(z)), 4.5, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = INK; ctx.lineWidth = 1.3; ctx.stroke();
      ctx.fillStyle = 'rgba(7,19,33,.5)';
      ctx.beginPath(); ctx.arc(px, Y(Math.max(0, z)), 3, 0, Math.PI * 2); ctx.fill();

      q('zv').textContent = z.toFixed(1);
      q('sv').textContent = silu(z).toFixed(3);
      q('rv').textContent = Math.max(0, z).toFixed(1);
      q('note').textContent = z < 0
        ? 'ReLU 在这里已经归零；SiLU 留了 ' + silu(z).toFixed(3) + ' 的小负开度——门没焊死。'
        : '开度 ' + silu(z).toFixed(3) + '：内容按这个比例放行，1 就是原样通过。';
    }

    q('z').addEventListener('input', draw);
    var onResize = function () { draw(); };
    window.addEventListener('resize', onResize);
    draw();
    window.setTimeout(draw, 60);

    return function dispose() {
      window.removeEventListener('resize', onResize);
    };
  }

  window.DLWidget.register('swiglu-gate', mount);
})();
