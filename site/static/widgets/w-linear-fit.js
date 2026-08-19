/* linear-fit — 拖动 a 与 b，看直线、残差、MSE 和两个偏导的变化。 */
(function () {
  'use strict';

  var DATA = [
    [0, 2.9], [0.5, 4.1], [1, 5.4], [1.5, 6.8], [2, 8.5], [2.5, 9.6], [3, 11.2],
  ];
  var A_RANGE = [-1, 5], B_RANGE = [-2, 8];
  var INK = '#071321', BLUE = '#0b63f3', RED = '#c0392b', PAPER = '#f7f5ef';

  function mean(xs) { return xs.reduce(function (s, v) { return s + v; }, 0) / xs.length; }

  function optimum() {
    var mx = mean(DATA.map(function (p) { return p[0]; }));
    var my = mean(DATA.map(function (p) { return p[1]; }));
    var num = 0, den = 0;
    DATA.forEach(function (p) { num += (p[0] - mx) * (p[1] - my); den += (p[0] - mx) * (p[0] - mx); });
    var a = num / den;
    return { a: a, b: my - a * mx };
  }

  function loss(a, b) {
    var s = 0;
    DATA.forEach(function (p) { var e = p[1] - (a * p[0] + b); s += e * e; });
    return s / DATA.length;
  }

  function grads(a, b) {
    var ga = 0, gb = 0;
    DATA.forEach(function (p) {
      var e = p[1] - (a * p[0] + b);
      ga += -2 * p[0] * e; gb += -2 * e;
    });
    return { a: ga / DATA.length, b: gb / DATA.length };
  }

  function mount(container) {
    container.innerHTML =
      '<p class="wg-title">拟合一条直线</p>' +
      '<p class="wg-sub">拖动滑杆调整 $a$（斜率）与 $b$（截距）…</p>'.replace(/\$(a|b)\$/g, '<b>$1</b>') +
      '<canvas class="wg-canvas" data-role="scatter" height="240"></canvas>' +
      '<div class="wg-block">' +
        '<div class="wg-label"><span>斜率 a</span><span data-role="aval"></span></div>' +
        '<input class="wg-slider" type="range" data-role="a" min="-1" max="5" step="0.02" value="0.5">' +
        '<div class="wg-label"><span>截距 b</span><span data-role="bval"></span></div>' +
        '<input class="wg-slider" type="range" data-role="b" min="-2" max="8" step="0.05" value="7.5">' +
      '</div>' +
      '<div class="wg-readout">' +
        '<div class="wg-stat"><span>MSE 损失</span><b data-role="mse"></b></div>' +
        '<div class="wg-stat"><span>∂L/∂a</span><b data-role="ga"></b></div>' +
        '<div class="wg-stat"><span>∂L/∂b</span><b data-role="gb"></b></div>' +
      '</div>' +
      '<p class="wg-note" data-role="hint"></p>' +
      '<div class="wg-controls">' +
        '<button type="button" class="wg-button is-primary" data-role="step">一步梯度下降</button>' +
        '<button type="button" class="wg-button" data-role="auto">自动下降</button>' +
        '<button type="button" class="wg-button" data-role="best">看最优解</button>' +
        '<button type="button" class="wg-button" data-role="reset">重置</button>' +
      '</div>' +
      '<div class="wg-label"><span>损失曲面 L(a, b)</span><span>● 当前 ★ 最优</span></div>' +
      '<canvas class="wg-canvas" data-role="contour" height="210"></canvas>' +
      '<p class="wg-note">曲面上每一点对应一组 (a, b)；颜色越深损失越小。梯度下降每步沿该点最陡的下坡方向走 η=0.12。</p>';

    var state = { a: 0.5, b: 7.5, timer: null };
    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };
    var scatter = q('scatter'), contour = q('contour');
    var best = optimum();

    function dpr(canvas) {
      var logicalHeight = parseFloat(canvas.getAttribute('height'));
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

    function drawScatter() {
      var c = dpr(scatter), ctx = c.ctx;
      ctx.clearRect(0, 0, c.w, c.h);
      var xMin = -0.3, xMax = 3.4, yMin = -0.5, yMax = 13;
      var sx = function (x) { return 34 + (x - xMin) / (xMax - xMin) * (c.w - 46); };
      var sy = function (y) { return c.h - 26 - (y - yMin) / (yMax - yMin) * (c.h - 40); };
      ctx.strokeStyle = 'rgba(7,19,33,.15)'; ctx.lineWidth = 1;
      for (var gx = 0; gx <= 3; gx += 1) { ctx.beginPath(); ctx.moveTo(sx(gx), sy(yMin)); ctx.lineTo(sx(gx), sy(yMax)); ctx.stroke(); }
      ctx.strokeStyle = INK;
      ctx.beginPath(); ctx.moveTo(sx(xMin), sy(0)); ctx.lineTo(sx(xMax), sy(0)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(sx(0), sy(yMin)); ctx.lineTo(sx(0), sy(yMax)); ctx.stroke();
      ctx.fillStyle = '#65717c'; ctx.font = '10px ui-monospace, monospace';
      ctx.fillText('x', sx(xMax) - 6, sy(0) + 14); ctx.fillText('y', sx(0) + 6, sy(yMax) + 4);
      // Residuals first so the line sits on top.
      ctx.strokeStyle = RED; ctx.setLineDash([3, 3]);
      DATA.forEach(function (p) {
        ctx.beginPath(); ctx.moveTo(sx(p[0]), sy(p[1])); ctx.lineTo(sx(p[0]), sy(state.a * p[0] + state.b)); ctx.stroke();
      });
      ctx.setLineDash([]);
      // Best line as a faint reference.
      ctx.strokeStyle = 'rgba(11,99,243,.28)';
      ctx.beginPath(); ctx.moveTo(sx(xMin), sy(best.a * xMin + best.b)); ctx.lineTo(sx(xMax), sy(best.a * xMax + best.b)); ctx.stroke();
      // Current line.
      ctx.strokeStyle = INK; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(sx(xMin), sy(state.a * xMin + state.b)); ctx.lineTo(sx(xMax), sy(state.a * xMax + state.b)); ctx.stroke();
      ctx.lineWidth = 1;
      DATA.forEach(function (p) {
        ctx.fillStyle = BLUE;
        ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), 3.4, 0, Math.PI * 2); ctx.fill();
      });
    }

    function drawContour() {
      var c = dpr(contour), ctx = c.ctx;
      var off = document.createElement('canvas');
      var N = 90;
      off.width = N; off.height = N;
      var octx = off.getContext('2d');
      var img = octx.createImageData(N, N);
      var lo = Math.log(loss(best.a, best.b) + 1e-9);
      var hi = Math.log(loss(A_RANGE[0], B_RANGE[1]) + 1e-9);
      for (var i = 0; i < N; i++) {
        for (var j = 0; j < N; j++) {
          var a = A_RANGE[0] + (A_RANGE[1] - A_RANGE[0]) * i / (N - 1);
          var b = B_RANGE[1] - (B_RANGE[1] - B_RANGE[0]) * j / (N - 1);
          var t = (Math.log(loss(a, b) + 1e-9) - lo) / (hi - lo);
          t = Math.max(0, Math.min(1, t));
          var k = (j * N + i) * 4;
          img.data[k] = Math.round(10 + 90 * t);
          img.data[k + 1] = Math.round(35 + 110 * t);
          img.data[k + 2] = Math.round(55 + 150 * t);
          img.data[k + 3] = 255;
        }
      }
      octx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.clearRect(0, 0, c.w, c.h);
      ctx.drawImage(off, 0, 0, c.w, c.h);
      var px = function (a) { return (a - A_RANGE[0]) / (A_RANGE[1] - A_RANGE[0]) * c.w; };
      var py = function (b) { return c.h - (b - B_RANGE[0]) / (B_RANGE[1] - B_RANGE[0]) * c.h; };
      ctx.strokeStyle = '#fffef9'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(px(state.a), py(state.b), 5, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = '#c8ff47';
      ctx.font = '13px ui-monospace, monospace';
      ctx.fillText('★', px(best.a) - 6, py(best.b) + 5);
      ctx.fillStyle = '#fffef9'; ctx.font = '10px ui-monospace, monospace';
      ctx.fillText('a →', c.w - 30, c.h - 6);
      ctx.fillText('b ↑', 6, 12);
    }

    function render() {
      q('a').value = String(state.a);
      q('b').value = String(state.b);
      q('aval').textContent = 'a = ' + state.a.toFixed(2);
      q('bval').textContent = 'b = ' + state.b.toFixed(2);
      var g = grads(state.a, state.b);
      q('mse').textContent = loss(state.a, state.b).toFixed(3);
      q('ga').textContent = g.a.toFixed(2);
      q('gb').textContent = g.b.toFixed(2);
      q('hint').textContent = g.a < 0 ? '∂L/∂a < 0：增大 a 会降低损失。' : '∂L/∂a > 0：减小 a 会降低损失。';
      drawScatter();
      drawContour();
    }

    function clamp(v, range) { return Math.max(range[0], Math.min(range[1], v)); }

    q('a').addEventListener('input', function () { state.a = parseFloat(this.value); render(); });
    q('b').addEventListener('input', function () { state.b = parseFloat(this.value); render(); });
    q('step').addEventListener('click', function () {
      var g = grads(state.a, state.b);
      state.a = clamp(state.a - 0.12 * g.a, A_RANGE);
      state.b = clamp(state.b - 0.12 * g.b, B_RANGE);
      render();
    });
    q('auto').addEventListener('click', function () {
      if (state.timer) {
        window.clearInterval(state.timer); state.timer = null;
        this.textContent = '自动下降'; return;
      }
      var button = this;
      button.textContent = '暂停';
      state.timer = window.setInterval(function () {
        var g = grads(state.a, state.b);
        state.a = clamp(state.a - 0.12 * g.a, A_RANGE);
        state.b = clamp(state.b - 0.12 * g.b, B_RANGE);
        render();
        if (Math.abs(g.a) < 0.004 && Math.abs(g.b) < 0.004) {
          window.clearInterval(state.timer); state.timer = null;
          button.textContent = '自动下降';
        }
      }, 220);
    });
    q('best').addEventListener('click', function () { state.a = best.a; state.b = best.b; render(); });
    q('reset').addEventListener('click', function () { state.a = 0.5; state.b = 7.5; render(); });

    var onResize = function () { render(); };
    window.addEventListener('resize', onResize);
    render();
    return function dispose() {
      if (state.timer) { window.clearInterval(state.timer); }
      window.removeEventListener('resize', onResize);
    };
  }

  window.DLWidget.register('linear-fit', mount);
})();
