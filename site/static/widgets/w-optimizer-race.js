/* optimizer-race — 五个优化器在同一个峡谷损失上赛跑：同样的 lr，不同的走法。 */
(function () {
  'use strict';

  var START = [-2.6, 2.2];
  var VX0 = -3.05, VX1 = 1.95, VY0 = -2.45, VY1 = 2.65;
  var XSPAN = VX1 - VX0, YSPAN = VY1 - VY0;
  var MAX_STEPS = 600, G_TOL = 1e-3, DIV_BOUND = 60, STEP_MS = 100;
  var EPS = 1e-8;
  var INK = '#071321', PAPER = '#fffef9', GREEN = '#c8ff47', RED = '#c0392b', GRAY = '#65717c';

  function loss(x, y) {
    return 0.6 * x * x + 3.5 * y * y + 0.9 * Math.sin(2.2 * x) * Math.cos(1.8 * y);
  }
  function grad(x, y) {
    return [1.2 * x + 1.98 * Math.cos(2.2 * x) * Math.cos(1.8 * y),
            7 * y - 1.62 * Math.sin(2.2 * x) * Math.sin(1.8 * y)];
  }
  function gnorm(x, y) { var g = grad(x, y); return Math.sqrt(g[0] * g[0] + g[1] * g[1]); }

  // 损失面固定，最小值用 Adam 从多个起点多跑一些步求出（约 -0.66 附近）。
  var MIN = (function () {
    var best = null;
    [[0, 0], [-1, 0.5], [1, -0.5], [-2, 0], [1.8, 0]].forEach(function (s0) {
      var x = s0[0], y = s0[1], m = [0, 0], v = [0, 0];
      for (var t = 1; t <= 4000; t++) {
        var g = grad(x, y);
        for (var i = 0; i < 2; i++) {
          m[i] = 0.9 * m[i] + 0.1 * g[i];
          v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
          var mh = m[i] / (1 - Math.pow(0.9, t)), vh = v[i] / (1 - Math.pow(0.999, t));
          var np = (i ? y : x) - 0.1 * mh / (Math.sqrt(vh) + 1e-8);
          if (i) { y = np; } else { x = np; }
        }
      }
      if (!best || loss(x, y) < best.f) { best = { x: x, y: y, f: loss(x, y) }; }
    });
    return best;
  })();

  function freshOpts() {
    return [
      { kind: 'sgd',      name: 'SGD',      color: '#071321', pos: START.slice(), trail: [START.slice()], f: loss(START[0], START[1]), done: null },
      { kind: 'mom',      name: 'MOMENTUM', color: '#0b63f3', pos: START.slice(), trail: [START.slice()], f: loss(START[0], START[1]), done: null, v: [0, 0] },
      { kind: 'adagrad',  name: 'ADAGRAD',  color: '#b45309', pos: START.slice(), trail: [START.slice()], f: loss(START[0], START[1]), done: null, G: [0, 0] },
      { kind: 'rmsprop',  name: 'RMSPROP',  color: '#7c3aed', pos: START.slice(), trail: [START.slice()], f: loss(START[0], START[1]), done: null, s: [0, 0] },
      { kind: 'adam',     name: 'ADAM',     color: '#0d8a6d', pos: START.slice(), trail: [START.slice()], f: loss(START[0], START[1]), done: null, m: [0, 0], v: [0, 0], t: 0 }
    ];
  }

  function mount(container) {
    container.innerHTML =
      '<style>' +
      '.or-layout{display:flex;gap:12px;align-items:stretch;margin:6px 0 2px}' +
      '.or-legend{flex:0 0 178px;display:flex;flex-direction:column;gap:5px}' +
      '.or-statbox{border:1px solid #071321;background:#f4f2ec;padding:7px 9px}' +
      '.or-statbox .lb{font:700 9px ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.08em;color:#65717c;text-transform:uppercase}' +
      '.or-statbox b{display:block;font:800 20px ui-monospace,SFMono-Regular,Consolas,monospace;color:#071321;line-height:1.15}' +
      '.or-statbox .sub{display:block;font:700 9.5px ui-monospace,SFMono-Regular,Consolas,monospace;color:#45515c}' +
      '.or-rows{display:flex;flex-direction:column;gap:5px}' +
      '.or-row{border:1px solid rgba(7,19,33,.3);border-left:3px solid #071321;background:#fffef9;padding:4px 7px 5px}' +
      '.or-row.is-arrive{background:#f2fbd9;border-color:#5c8a2a;border-left-color:#5c8a2a}' +
      '.or-row.is-div{background:#f9ece9;border-color:#c0392b;border-left-color:#c0392b}' +
      '.or-row.is-cap{opacity:.72}' +
      '.or-top{display:flex;align-items:center;gap:6px}' +
      '.or-chip{width:10px;height:10px;border:1px solid #071321;flex:none}' +
      '.or-name{font:700 10px ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.04em;color:#071321}' +
      '.or-tag{margin-left:auto;font:700 9px ui-monospace,SFMono-Regular,Consolas,monospace;padding:1px 4px;border:1px solid rgba(7,19,33,.3);color:#65717c;background:#f4f2ec;white-space:nowrap}' +
      '.or-tag.is-arrive{background:#c8ff47;border-color:#071321;color:#071321}' +
      '.or-tag.is-div{background:#c0392b;border-color:#071321;color:#fffef9}' +
      '.or-val{font:700 10px ui-monospace,SFMono-Regular,Consolas,monospace;color:#263541;margin-top:2px;white-space:nowrap}' +
      '.or-fval{font:700 10px ui-monospace,SFMono-Regular,Consolas,monospace;color:#65717c;white-space:nowrap}' +
      '.or-rule{font:700 9px ui-monospace,SFMono-Regular,Consolas,monospace;line-height:1.7;color:#65717c;border-top:1px dashed rgba(7,19,33,.3);padding-top:6px}' +
      '.or-stage{flex:1;min-width:0}' +
      '.or-hint{font:700 10px ui-monospace,SFMono-Regular,Consolas,monospace;color:#65717c;align-self:center;margin-left:2px}' +
      '@media (max-width:600px){.or-layout{flex-direction:column}.or-legend{flex:auto}.or-rows{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}}' +
      '</style>' +
      '<p class="wg-title">优化器赛跑：同一起点，五种走法</p>' +
      '<p class="wg-sub">f(x,y) = 0.6x² + 3.5y² + 0.9·sin(2.2x)·cos(1.8y)，五个优化器同起点、同 lr 赛跑。</p>' +
      '<div class="or-layout">' +
        '<div class="or-legend">' +
          '<div class="or-statbox"><span class="lb">全局步数 t</span><b data-role="t">0</b><span class="sub" data-role="sub">到达 0 · 发散 0</span></div>' +
          '<div class="or-rows" data-role="rows"></div>' +
          '<div class="or-rule">停止条件 ‖∇f‖ &lt; 1e-3<br>越界视为发散 · 上限 600 步<br>起点 (-2.6, 2.2)，f = ' + loss(START[0], START[1]).toFixed(2) + '</div>' +
        '</div>' +
        '<div class="or-stage"><canvas class="wg-canvas" data-role="cv" height="440"></canvas></div>' +
      '</div>' +
      '<div class="wg-block">' +
        '<div class="wg-label"><span>学习率 lr — 五个优化器共用</span><span data-role="lrval">0.25</span></div>' +
        '<input class="wg-slider" type="range" data-role="lr" min="0.01" max="0.6" step="0.01" value="0.25">' +
        '<div class="wg-label"><span>β = MOMENTUM / ρ = RMSPROP</span><span data-role="betaval">0.90</span></div>' +
        '<input class="wg-slider" type="range" data-role="beta" min="0.5" max="0.99" step="0.01" value="0.90">' +
      '</div>' +
      '<div class="wg-controls">' +
        '<button type="button" class="wg-button is-primary" data-role="play">播放</button>' +
        '<button type="button" class="wg-button" data-role="step">单步</button>' +
        '<button type="button" class="wg-button" data-role="reset">重置</button>' +
        '<span class="or-hint">拖动 lr / β 立即生效</span>' +
      '</div>' +
      '<p class="wg-note">lr 拉到 0.3 以上可见 SGD 发散。</p>';

    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };
    var cv = q('cv');
    var rowsBox = q('rows');

    // 图例行
    var opts = freshOpts();
    var rowRefs = opts.map(function (o) {
      var row = document.createElement('div');
      row.className = 'or-row';
      row.style.borderLeftColor = o.color;
      row.innerHTML =
        '<div class="or-top"><span class="or-chip" style="background:' + o.color + '"></span>' +
        '<span class="or-name">' + o.name + '</span><span class="or-tag" data-tag>在跑</span></div>' +
        '<div class="or-val" data-pos></div>' +
        '<div class="or-fval" data-f></div>';
      rowsBox.appendChild(row);
      return { row: row, tag: row.querySelector('[data-tag]'), pos: row.querySelector('[data-pos]'), f: row.querySelector('[data-f]') };
    });

    var state = { t: 0, lr: 0.25, beta: 0.9, timer: null, boot: null };
    var bg = null;

    function allDone() { return opts.every(function (o) { return o.done; }); }

    function clearBoot() {
      if (state.boot) { window.clearTimeout(state.boot); state.boot = null; }
    }

    function setPlaying(on) {
      if (on) {
        if (state.timer) { return; }
        state.timer = window.setInterval(stepOnce, STEP_MS);
        q('play').textContent = '暂停';
      } else {
        if (state.timer) { window.clearInterval(state.timer); state.timer = null; }
        q('play').textContent = allDone() ? '重新播放' : '播放';
      }
    }

    function resetState() {
      clearBoot();
      setPlaying(false);
      state.t = 0;
      opts = freshOpts();
      opts.forEach(function (o, k) {
        rowRefs[k].row.className = 'or-row';
        rowRefs[k].row.style.borderLeftColor = o.color;
      });
      render();
    }

    function stepOnce() {
      if (state.t >= MAX_STEPS) { setPlaying(false); return; }
      var lr = state.lr, be = state.beta;
      state.t += 1;
      opts.forEach(function (o) {
        if (o.done) { return; }
        var g = grad(o.pos[0], o.pos[1]);
        var nx, ny;
        if (o.kind === 'sgd') {
          nx = o.pos[0] - lr * g[0]; ny = o.pos[1] - lr * g[1];
        } else if (o.kind === 'mom') {
          o.v[0] = be * o.v[0] + g[0]; o.v[1] = be * o.v[1] + g[1];
          nx = o.pos[0] - lr * o.v[0]; ny = o.pos[1] - lr * o.v[1];
        } else if (o.kind === 'adagrad') {
          o.G[0] += g[0] * g[0]; o.G[1] += g[1] * g[1];
          nx = o.pos[0] - lr * g[0] / (Math.sqrt(o.G[0]) + EPS);
          ny = o.pos[1] - lr * g[1] / (Math.sqrt(o.G[1]) + EPS);
        } else if (o.kind === 'rmsprop') {
          o.s[0] = be * o.s[0] + (1 - be) * g[0] * g[0];
          o.s[1] = be * o.s[1] + (1 - be) * g[1] * g[1];
          nx = o.pos[0] - lr * g[0] / (Math.sqrt(o.s[0]) + EPS);
          ny = o.pos[1] - lr * g[1] / (Math.sqrt(o.s[1]) + EPS);
        } else {
          o.t += 1;
          var b1 = 0.9, b2 = 0.999, tt = o.t;
          o.m[0] = b1 * o.m[0] + (1 - b1) * g[0]; o.m[1] = b1 * o.m[1] + (1 - b1) * g[1];
          o.v[0] = b2 * o.v[0] + (1 - b2) * g[0] * g[0]; o.v[1] = b2 * o.v[1] + (1 - b2) * g[1] * g[1];
          var m0 = o.m[0] / (1 - Math.pow(b1, tt)), v0 = o.v[0] / (1 - Math.pow(b2, tt));
          var m1 = o.m[1] / (1 - Math.pow(b1, tt)), v1 = o.v[1] / (1 - Math.pow(b2, tt));
          nx = o.pos[0] - lr * m0 / (Math.sqrt(v0) + EPS);
          ny = o.pos[1] - lr * m1 / (Math.sqrt(v1) + EPS);
        }
        o.pos = [nx, ny];
        o.trail.push(o.pos);
        o.f = loss(nx, ny);
        if (!isFinite(nx) || !isFinite(ny) || Math.abs(nx) > DIV_BOUND || Math.abs(ny) > DIV_BOUND) {
          o.done = { type: 'div', step: state.t };
          return;
        }
        if (gnorm(nx, ny) < G_TOL) { o.done = { type: 'arrive', step: state.t }; }
      });
      if (state.t >= MAX_STEPS) {
        opts.forEach(function (o) { if (!o.done) { o.done = { type: 'cap' }; } });
      }
      if (allDone()) { setPlaying(false); }
      render();
    }

    function dpr(canvas, logicalHeight) {
      var cssWidth = canvas.parentElement ? canvas.parentElement.clientWidth : 0;
      cssWidth = Math.max(240, Math.min(cssWidth || 440, 470));
      var ratio = window.devicePixelRatio || 1;
      canvas.style.width = cssWidth + 'px';
      canvas.style.height = logicalHeight + 'px';
      canvas.width = Math.round(cssWidth * ratio);
      canvas.height = Math.round(logicalHeight * ratio);
      var ctx = canvas.getContext('2d');
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return { ctx: ctx, w: cssWidth, h: logicalHeight };
    }

    function mixColor(t) {
      var A = [250, 249, 243], B = [204, 215, 232], C = [121, 148, 187];
      var p = t < 0.5 ? [A, B, t * 2] : [B, C, (t - 0.5) * 2];
      return [Math.round(p[0][0] + (p[1][0] - p[0][0]) * p[2]),
              Math.round(p[0][1] + (p[1][1] - p[0][1]) * p[2]),
              Math.round(p[0][2] + (p[1][2] - p[0][2]) * p[2])];
    }

    // 等值线：marching squares，在一个低分辨率网格上插值出线段。
    function isoSegments(vals, nx, ny, L, w, h) {
      var segs = [];
      function ip(v0, v1) {
        var den = v1 - v0;
        var u = den ? (L - v0) / den : 0.5;
        return u < 0 ? 0 : (u > 1 ? 1 : u);
      }
      for (var j = 0; j < ny - 1; j++) {
        for (var i = 0; i < nx - 1; i++) {
          var a = vals[j][i], b = vals[j][i + 1], c = vals[j + 1][i + 1], d = vals[j + 1][i];
          var id = (a > L ? 8 : 0) | (b > L ? 4 : 0) | (c > L ? 2 : 0) | (d > L ? 1 : 0);
          if (id === 0 || id === 15) { continue; }
          var ax = i / (nx - 1) * w, bx = (i + 1) / (nx - 1) * w;
          var ay = j / (ny - 1) * h, by = (j + 1) / (ny - 1) * h;
          var eTx = ax + ip(a, b) * (bx - ax), eBx = ax + ip(d, c) * (bx - ax);
          var eLy = ay + ip(a, d) * (by - ay), eRy = ay + ip(b, c) * (by - ay);
          function seg(x1, y1, x2, y2) { segs.push(x1, y1, x2, y2); }
          switch (id) {
            case 1: case 14: seg(ax, eLy, eBx, by); break;
            case 2: case 13: seg(eBx, by, bx, eRy); break;
            case 3: case 12: seg(ax, eLy, bx, eRy); break;
            case 4: case 11: seg(eTx, ay, bx, eRy); break;
            case 6: case 9:  seg(eTx, ay, eBx, by); break;
            case 7: case 8:  seg(eTx, ay, ax, eLy); break;
            case 5:  seg(eTx, ay, ax, eLy); seg(eBx, by, bx, eRy); break;
            case 10: seg(eTx, ay, bx, eRy); seg(eBx, by, ax, eLy); break;
          }
        }
      }
      return segs;
    }

    function haloText(ctx, s, x, y, color) {
      ctx.font = '10px ui-monospace, monospace';
      ctx.lineJoin = 'round';
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(255,254,249,.92)';
      ctx.strokeText(s, x, y);
      ctx.fillStyle = color || INK;
      ctx.fillText(s, x, y);
    }

    // 背景（等高线热图 + 坐标轴 + 起点/极小值标记）只在尺寸变化时重建。
    function buildBackground(w, h) {
      var ratio = window.devicePixelRatio || 1;
      var off = document.createElement('canvas');
      off.width = Math.max(1, Math.round(w * ratio));
      off.height = Math.max(1, Math.round(h * ratio));
      var ctx = off.getContext('2d');
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

      var nx = 120, ny = Math.max(48, Math.round(nx * h / w));
      var vals = [], fmin = Infinity, fmax = -Infinity, i, j;
      for (j = 0; j < ny; j++) {
        var row = [], y = VY1 - j / (ny - 1) * YSPAN;
        for (i = 0; i < nx; i++) {
          var v = loss(VX0 + i / (nx - 1) * XSPAN, y);
          row.push(v);
          if (v < fmin) { fmin = v; }
          if (v > fmax) { fmax = v; }
        }
        vals.push(row);
      }

      var low = document.createElement('canvas');
      low.width = nx; low.height = ny;
      var lctx = low.getContext('2d');
      var img = lctx.createImageData(nx, ny);
      var lo = Math.log(0.05), hi = Math.log(fmax - fmin + 0.05);
      for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
          var t = (Math.log(vals[j][i] - fmin + 0.05) - lo) / (hi - lo);
          t = t < 0 ? 0 : (t > 1 ? 1 : t);
          var rgb = mixColor(t), k = (j * nx + i) * 4;
          img.data[k] = rgb[0]; img.data[k + 1] = rgb[1]; img.data[k + 2] = rgb[2]; img.data[k + 3] = 255;
        }
      }
      lctx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(low, 0, 0, w, h);

      ctx.strokeStyle = 'rgba(7,19,33,.17)';
      ctx.lineWidth = 1;
      for (var lv = 1; lv <= 9; lv++) {
        var L = fmin - 0.05 + Math.exp(lo + lv / 10 * (hi - lo));
        var segs = isoSegments(vals, nx, ny, L, w, h);
        ctx.beginPath();
        for (var s = 0; s < segs.length; s += 4) {
          ctx.moveTo(segs[s], segs[s + 1]);
          ctx.lineTo(segs[s + 2], segs[s + 3]);
        }
        ctx.stroke();
      }

      var X = function (x) { return (x - VX0) / XSPAN * w; };
      var Y = function (y) { return h - (y - VY0) / YSPAN * h; };

      // 坐标轴
      ctx.strokeStyle = 'rgba(7,19,33,.30)';
      ctx.beginPath(); ctx.moveTo(X(0), 0); ctx.lineTo(X(0), h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(w, Y(0)); ctx.stroke();
      ctx.fillStyle = GRAY; ctx.font = '10px ui-monospace, monospace';
      ctx.fillText('x →', w - 28, Y(0) - 5);
      ctx.fillText('y ↑', X(0) + 5, 11);

      // 起点
      var sx = X(START[0]), sy = Y(START[1]);
      ctx.fillStyle = INK;
      ctx.fillRect(sx - 3.5, sy - 3.5, 7, 7);
      ctx.strokeStyle = PAPER; ctx.lineWidth = 1.2;
      ctx.strokeRect(sx - 3.5, sy - 3.5, 7, 7);
      haloText(ctx, '起点 (-2.6, 2.2)', sx + 9, sy + 4);

      // 极小值
      var mx = X(MIN.x), my = Y(MIN.y), r = 5;
      ctx.beginPath();
      ctx.moveTo(mx, my - r); ctx.lineTo(mx + r, my); ctx.lineTo(mx, my + r); ctx.lineTo(mx - r, my);
      ctx.closePath();
      ctx.fillStyle = GREEN; ctx.fill();
      ctx.strokeStyle = INK; ctx.lineWidth = 1.2; ctx.stroke();
      haloText(ctx, '极小值 (' + MIN.x.toFixed(2) + ', 0)', mx - 14, my - 10);

      return { w: w, h: h, canvas: off };
    }

    function drawScene() {
      var cssW = Math.max(240, Math.min(cv.parentElement ? cv.parentElement.clientWidth : 440, 470));
      var logicalH = Math.round(cssW * YSPAN / XSPAN);
      var c = dpr(cv, logicalH), ctx = c.ctx;
      if (!bg || bg.w !== c.w || bg.h !== c.h) { bg = buildBackground(c.w, c.h); }
      ctx.clearRect(0, 0, c.w, c.h);
      ctx.drawImage(bg.canvas, 0, 0, c.w, c.h);

      var X = function (x) { return (x - VX0) / XSPAN * c.w; };
      var Y = function (y) { return c.h - (y - VY0) / YSPAN * c.h; };

      opts.forEach(function (o) {
        if (o.trail.length > 1) {
          ctx.lineJoin = 'round'; ctx.lineCap = 'round';
          ctx.beginPath();
          o.trail.forEach(function (p, i) {
            var px = X(p[0]), py = Y(p[1]);
            if (i === 0) { ctx.moveTo(px, py); } else { ctx.lineTo(px, py); }
          });
          ctx.strokeStyle = 'rgba(255,254,249,.7)'; ctx.lineWidth = 3.4; ctx.stroke();
          ctx.strokeStyle = o.color; ctx.lineWidth = 1.9; ctx.stroke();
        }
        var px = X(o.pos[0]), py = Y(o.pos[1]);
        if (px > -40 && px < c.w + 40 && py > -40 && py < c.h + 40) {
          if (o.done && o.done.type === 'arrive') {
            ctx.beginPath(); ctx.arc(px, py, 5.4, 0, 6.2832);
            ctx.fillStyle = GREEN; ctx.fill();
            ctx.strokeStyle = INK; ctx.lineWidth = 1.3; ctx.stroke();
          } else {
            ctx.beginPath(); ctx.arc(px, py, 3.8, 0, 6.2832);
            ctx.fillStyle = o.color; ctx.fill();
            ctx.strokeStyle = PAPER; ctx.lineWidth = 1.4; ctx.stroke();
          }
        }
      });
    }

    function fmt(v) { return (Math.abs(v) < 0.005 ? 0 : v).toFixed(2); }

    function render() {
      drawScene();
      var nArr = 0, nDiv = 0;
      opts.forEach(function (o, k) {
        var ref = rowRefs[k];
        ref.pos.textContent = '(' + fmt(o.pos[0]) + ', ' + fmt(o.pos[1]) + ')';
        ref.f.textContent = 'f = ' + o.f.toFixed(3);
        if (!o.done) {
          ref.row.className = 'or-row';
          ref.tag.className = 'or-tag';
          ref.tag.textContent = '在跑';
        } else if (o.done.type === 'arrive') {
          nArr += 1;
          ref.row.className = 'or-row is-arrive';
          ref.tag.className = 'or-tag is-arrive';
          ref.tag.textContent = '到达·' + o.done.step;
        } else if (o.done.type === 'div') {
          nDiv += 1;
          ref.row.className = 'or-row is-div';
          ref.tag.className = 'or-tag is-div';
          ref.tag.textContent = '发散·' + o.done.step;
        } else {
          ref.row.className = 'or-row is-cap';
          ref.tag.className = 'or-tag';
          ref.tag.textContent = '600步未到';
        }
        ref.row.style.borderLeftColor = o.color;
      });
      q('t').textContent = String(state.t);
      q('sub').textContent = '到达 ' + nArr + ' · 发散 ' + nDiv;
    }

    q('lr').addEventListener('input', function () {
      state.lr = parseFloat(this.value);
      q('lrval').textContent = state.lr.toFixed(2);
    });
    q('beta').addEventListener('input', function () {
      state.beta = parseFloat(this.value);
      q('betaval').textContent = state.beta.toFixed(2);
    });
    q('play').addEventListener('click', function () {
      clearBoot();
      if (state.timer) { setPlaying(false); return; }
      if (allDone()) { resetState(); }
      setPlaying(true);
    });
    q('step').addEventListener('click', function () {
      clearBoot();
      if (allDone()) { return; }
      stepOnce();
    });
    q('reset').addEventListener('click', function () { resetState(); });

    var onResize = function () { bg = null; render(); };
    window.addEventListener('resize', onResize);
    render();

    // 首次进入自动开跑，读者抬头就能看到比赛进行中。
    state.boot = window.setTimeout(function () {
      state.boot = null;
      if (!allDone()) { setPlaying(true); }
    }, 600);

    return function dispose() {
      clearBoot();
      if (state.timer) { window.clearInterval(state.timer); state.timer = null; }
      window.removeEventListener('resize', onResize);
    };
  }

  window.DLWidget.register('optimizer-race', mount);
})();
