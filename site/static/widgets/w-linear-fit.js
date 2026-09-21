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
      '<p class="wg-sub">七个无单位的教学点，用于观察直线与误差；不是薪资数据，也不是薪资数据的标准化结果。</p>' +
      '<p class="linear-data">(0,2.9)、(0.5,4.1)、(1,5.4)、(1.5,6.8)、(2,8.5)、(2.5,9.6)、(3,11.2)。黑线：当前预测；红虚线：误差；浅蓝线：最小二乘解。</p>' +
      '<canvas class="wg-canvas" data-role="scatter" height="240"></canvas>' +
      '<div class="wg-block">' +
        '<div class="wg-label"><span>斜率 a</span><span data-role="aval"></span></div>' +
        '<input class="wg-slider" aria-label="斜率 a" type="range" data-role="a" min="-1" max="5" step="0.02" value="0.5">' +
        '<div class="wg-label"><span>截距 b</span><span data-role="bval"></span></div>' +
        '<input class="wg-slider" aria-label="截距 b" type="range" data-role="b" min="-2" max="8" step="0.05" value="7.5">' +
      '</div>' +
      '<div class="wg-label"><span>学习率 η</span><span data-role="lrval">0.12</span></div>' +
      '<input class="wg-slider" aria-label="学习率" type="range" data-role="lr" min="0.01" max="1" step="0.01" value="0.12">' +
      '<label>更新哪些参数 <select data-role="mode" aria-label="更新哪些参数"><option value="both">同时更新 a 和 b</option><option value="a">只更新 a</option><option value="b">只更新 b</option></select></label>' +
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
      '<p class="linear-comparison" data-role="comparison" aria-live="polite">尚未更新。先猜方向，再点一步。</p>' +
      '<details data-role="surface"><summary>进阶观察：参数平面上的损失</summary>' +
      '<div class="wg-label"><span>损失曲面 L(a, b)</span><span>● 当前 ★ 最优</span></div>' +
      '<canvas class="wg-canvas" data-role="contour" height="210"></canvas>' +
      '<p class="wg-note">曲面上每一点对应一组 (a, b)；颜色越深损失越小。这里画出固定范围 a∈[−1,5]、b∈[−2,8]。若参数走出范围，当前点会离开图面；上方数值仍显示真实更新结果。</p></details>';

    var state = { a: 0.5, b: 7.5, lr: 0.12, mode: "both", timer: null };
    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };
    var scatter = q('scatter'), contour = q('contour');
    var best = optimum();
    // Logical sizes are constants — never read back from the width/height
    // attributes, which this component rewrites with device-pixel values
    // (reading them back made every re-render double the canvas height).
    var SCATTER_H = 240, CONTOUR_H = 210;

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

    function drawScatter() {
      var c = dpr(scatter, SCATTER_H), ctx = c.ctx;
      ctx.clearRect(0, 0, c.w, c.h);
      var xMin = -0.3, xMax = 3.4;
      var ends = [state.b, state.a * 3 + state.b];
      var yMin = Math.min(-0.5, ends[0] - 1, ends[1] - 1), yMax = Math.max(13, ends[0] + 1, ends[1] + 1);
      var sx = function (x) { return 34 + (x - xMin) / (xMax - xMin) * (c.w - 46); };
      var sy = function (y) { return c.h - 26 - (y - yMin) / (yMax - yMin) * (c.h - 40); };
      ctx.strokeStyle = 'rgba(7,19,33,.15)'; ctx.lineWidth = 1;
      for (var gx = 0; gx <= 3; gx += 1) { ctx.beginPath(); ctx.moveTo(sx(gx), sy(yMin)); ctx.lineTo(sx(gx), sy(yMax)); ctx.stroke(); }
      ctx.strokeStyle = INK;
      ctx.beginPath(); ctx.moveTo(sx(xMin), sy(0)); ctx.lineTo(sx(xMax), sy(0)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(sx(0), sy(yMin)); ctx.lineTo(sx(0), sy(yMax)); ctx.stroke();
      ctx.fillStyle = '#65717c'; ctx.font = '10px ui-monospace, monospace';
      ctx.fillText('x', sx(xMax) - 6, c.h - 5); ctx.fillText('y', 6, 12);
      for (var tx = 0; tx <= 3; tx++) { ctx.fillText(String(tx), sx(tx) - 3, c.h - 9); }
      for (var tick = 0; tick <= 4; tick++) {
        var yTick = yMin + (yMax - yMin) * tick / 4;
        ctx.fillText(Math.abs(yTick) >= 100 ? yTick.toExponential(0) : yTick.toFixed(1), 1, sy(yTick) + 3);
      }
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
      var c = dpr(contour, CONTOUR_H), ctx = c.ctx;
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
      ['a', 'b'].forEach(function (key) {
        var range = key === 'a' ? A_RANGE : B_RANGE;
        q(key).min = String(Math.min(range[0], Math.floor(state[key])));
        q(key).max = String(Math.max(range[1], Math.ceil(state[key])));
      });
      q('lrval').textContent = state.lr.toFixed(2);
      q('a').value = String(state.a);
      q('b').value = String(state.b);
      q('aval').textContent = 'a = ' + state.a.toFixed(2);
      q('bval').textContent = 'b = ' + state.b.toFixed(2);
      var g = grads(state.a, state.b);
      q('mse').textContent = loss(state.a, state.b).toFixed(3);
      q('ga').textContent = g.a.toFixed(2);
      q('gb').textContent = g.b.toFixed(2);
      var key = state.mode === 'b' ? 'b' : 'a';
      q('hint').textContent = Math.abs(g[key]) < 0.0001 ? '∂L/∂' + key + ' 接近 0：局部一阶变化很小。' :
        '∂L/∂' + key + (g[key] < 0 ? ' < 0：小幅增大 ' : ' > 0：小幅减小 ') + key + ' 有助于降低损失；大步更新不保证下降。';
      drawScatter();
      drawContour();
    }

    function stop() {
      if (state.timer) { window.clearInterval(state.timer); state.timer = null; }
      q('auto').textContent = '自动下降';
    }
    function step() {
      var before = { a: state.a, b: state.b, loss: loss(state.a, state.b) };
      var g = grads(before.a, before.b);
      var nextA = before.a - (state.mode === 'b' ? 0 : state.lr * g.a);
      var nextB = before.b - (state.mode === 'a' ? 0 : state.lr * g.b);
      if (!Number.isFinite(nextA + nextB) || Math.max(Math.abs(nextA), Math.abs(nextB)) > 100000) {
        stop();
        q('comparison').textContent = '更新会让参数绝对值超过 100000，已停止，保留上一步结果。损失持续增大时可以减小学习率并重置再比较。';
        return;
      }
      state.a = nextA; state.b = nextB;
      var after = loss(state.a, state.b);
      q('comparison').textContent = 'η=' + state.lr.toFixed(2) + '；a：' + before.a.toFixed(3) + ' → ' + state.a.toFixed(3) +
        '，b：' + before.b.toFixed(3) + ' → ' + state.b.toFixed(3) + '。使用更新前梯度 (' + g.a.toFixed(3) + ', ' + g.b.toFixed(3) +
        ')；损失：' + before.loss.toFixed(3) + ' → ' + after.toFixed(3) + (after > before.loss ? '，这一步损失增大，步长可能过大。' : '，这一步损失没有增大。');
      render();
      var movingA = state.mode !== 'b' && Math.abs(g.a) >= 0.004;
      var movingB = state.mode !== 'a' && Math.abs(g.b) >= 0.004;
      if (!movingA && !movingB) { stop(); }
    }
    ['a', 'b', 'lr'].forEach(function (key) {
      q(key).addEventListener('input', function () {
        stop(); state[key] = parseFloat(this.value);
        q('comparison').textContent = '已手动调整参数或学习率。下一步会从当前显示的数值开始。'; render();
      });
    });
    q('mode').addEventListener('change', function () { stop(); state.mode = this.value; render(); });
    q('step').addEventListener('click', function () { stop(); step(); });
    q('auto').addEventListener('click', function () {
      if (state.timer) { stop(); return; }
      this.textContent = '暂停'; state.timer = window.setInterval(step, 350);
    });
    q('best').addEventListener('click', function () {
      stop(); state.a = best.a; state.b = best.b;
      q('comparison').textContent = '显示这七个点的最小二乘解，不是一次梯度更新。'; render();
    });
    q('reset').addEventListener('click', function () {
      stop(); state.a = 0.5; state.b = 7.5;
      q('comparison').textContent = '已回到相同起点 a=0.5、b=7.5，保留当前学习率与更新方式，方便对照。'; render();
    });
    q('surface').addEventListener('toggle', function () { if (this.open) { drawContour(); } });

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
