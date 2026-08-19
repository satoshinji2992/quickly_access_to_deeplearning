/* token-embed-3d — 20 个 token 的 8 维嵌入向量压成 3 维散点：拖拽旋转、点击查 8 维原向量。
 * 数据资产：assets/widgets/token_embeddings.json（tokens/xyz/clusters/vec8/note）。
 * 簇配色按首次出现顺序固定分配，已用 OKLab 全对 CVD 校验通过。 */
(function () {
  'use strict';

  var INK = '#071321', PAPER = '#f7f5ef', BLUE = '#0b63f3', RED = '#c0392b',
      NEON = '#c8ff47', MUTED = '#65717c', INKSOFT = 'rgba(7,19,33,.55)';
  var PALETTE = ['#0b63f3', '#d95926', '#0e8a6d', '#4a3aa7'];
  var DATA_URL = '/quickly_access_to_deeplearning/assets/widgets/token_embeddings.json';
  var MONO = 'ui-monospace, SFMono-Regular, Consolas, monospace';

  var STYLE_TEXT =
    '.te3-main{display:flex;gap:14px;align-items:stretch;margin:6px 0 0}' +
    '.te3-left{flex:1 1 auto;min-width:0}' +
    '.te3-side{flex:0 0 196px;border:1px solid rgba(7,19,33,.3);background:#f4f2ec;padding:10px}' +
    '.te3-wrap{position:relative}' +
    '.te3-wrap canvas{touch-action:none;cursor:grab}' +
    '.te3-wrap canvas:focus-visible{outline:2px solid ' + BLUE + ';outline-offset:2px}' +
    '.te3-tip{position:absolute;z-index:2;pointer-events:none;white-space:nowrap;' +
      'background:#071321;color:#fffef9;padding:3px 7px;font:700 10px ' + MONO + '}' +
    '.te3-legend{display:flex;flex-wrap:wrap;gap:5px 14px;margin-top:9px;' +
      'font:700 10px ' + MONO + ';letter-spacing:.06em;text-transform:uppercase;color:#45515c}' +
    '.te3-key{display:inline-flex;align-items:center;gap:5px}' +
    '.te3-key i{width:10px;height:10px;border:1px solid #071321;flex:0 0 auto}' +
    '.te3-tok{display:flex;align-items:center;gap:9px;margin-bottom:4px}' +
    '.te3-glyph{display:grid;place-items:center;width:30px;height:30px;flex:0 0 auto;' +
      'border:1px solid #071321;color:#fffef9;font:800 15px ' + MONO + '}' +
    '.te3-meta b{display:block;font:800 14px ' + MONO + ';color:#071321}' +
    '.te3-meta span{font:700 9px ' + MONO + ';letter-spacing:.08em;color:#65717c;text-transform:uppercase}' +
    '.te3-coord{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}' +
    '.te3-coord div{border:1px solid rgba(7,19,33,.25);background:#fffef9;padding:4px 5px;text-align:center}' +
    '.te3-coord b{display:block;font:800 12px ' + MONO + ';color:#071321}' +
    '.te3-coord span{font:700 8.5px ' + MONO + ';letter-spacing:.08em;color:#65717c;text-transform:uppercase}' +
    '.te3-empty{font-size:11px;color:#65717c;border:1px dashed rgba(7,19,33,.35);' +
      'padding:10px;background:#fffef9}' +
    '.te3-err{border:1px solid #c0392b;background:rgba(192,57,43,.07);color:#c0392b;' +
      'padding:10px 12px;font:600 11px ' + MONO + ';line-height:1.7}' +
    '.te3-err code{font:inherit;word-break:break-all}' +
    '.te3-barswrap{margin:2px 0 4px}' +
    '@media (max-width:620px){.te3-main{flex-direction:column}.te3-side{flex:auto}}';

  /* 校验数据形状；失败时返回 {err}，由 showError 呈现。 */
  function parseData(raw) {
    if (!raw || typeof raw !== 'object') { return { err: 'JSON 结构不是对象' }; }
    var tokens = raw.tokens, xyz = raw.xyz, vec8 = raw.vec8, clusters = raw.clusters;
    if (!Array.isArray(tokens) || !Array.isArray(xyz) || !Array.isArray(vec8) ||
        !Array.isArray(clusters)) {
      return { err: '缺少 tokens/xyz/clusters/vec8 数组字段' };
    }
    var n = tokens.length;
    if (n === 0 || xyz.length !== n || vec8.length !== n || clusters.length !== n) {
      return { err: '数组长度不一致：tokens=' + n + ' xyz=' + xyz.length +
        ' vec8=' + vec8.length + ' clusters=' + clusters.length };
    }
    var pts = [], vecs = [], names = [];
    for (var i = 0; i < n; i++) {
      var p = xyz[i], v = vec8[i];
      if (!Array.isArray(p) || p.length !== 3 || !Array.isArray(v) || v.length !== 8 ||
          !p.every(isFinite) || !v.every(isFinite)) {
        return { err: '第 ' + i + ' 个 token 的 xyz/vec8 形状或数值不合法' };
      }
      pts.push([+p[0], +p[1], +p[2]]);
      vecs.push(v.slice());
      var c = clusters[i];
      names.push(typeof c === 'number' ? '簇 ' + (c + 1) : String(c));
    }
    // 簇按首次出现顺序编号：配色固定，不随选中状态重排。
    var uniq = [], indexOf = {};
    names.forEach(function (nm) {
      if (!(nm in indexOf)) { indexOf[nm] = uniq.length; uniq.push(nm); }
    });
    return {
      tokens: tokens.map(String), xyz: pts, vec8: vecs, names: names,
      uniq: uniq, indexOf: indexOf, note: raw.note ? String(raw.note) : '',
    };
  }

  function mount(container) {
    var styleEl = document.createElement('style');
    styleEl.textContent = STYLE_TEXT;
    var disposed = false;
    var innerDispose = null;

    // innerHTML 重置后私有 <style> 会丢失，统一在这里挂回。
    function setBody(html) {
      container.innerHTML = html;
      container.appendChild(styleEl);
    }

    function showError(msg) {
      setBody(
        '<p class="wg-title">Token 嵌入：8 维 → 3 维投影</p>' +
        '<p class="te3-err">向量数据加载失败：' + msg + '<br>' +
        '数据文件应位于 <code>' + DATA_URL + '</code>（由本地脚本生成）。' +
        '请确认文件存在后刷新页面。</p>');
    }

    setBody(
      '<p class="wg-title">Token 嵌入：8 维 → 3 维投影</p>' +
      '<p class="wg-sub">正在加载 token 向量数据…</p>');

    fetch(DATA_URL).then(function (r) {
      if (!r.ok) { throw new Error('HTTP ' + r.status); }
      return r.json();
    }).then(function (raw) {
      if (disposed) { return; }
      var d = parseData(raw);
      if (d.err) { showError(d.err); return; }
      innerDispose = build(container, d, setBody);
    }).catch(function (e) {
      if (!disposed) { showError(e && e.message ? e.message : String(e)); }
    });

    return function dispose() {
      disposed = true;
      if (innerDispose) { innerDispose(); }
    };
  }

  function build(container, D, setBody) {
    setBody(
      '<p class="wg-title">Token 嵌入：8 维 → 3 维投影</p>' +
      '<p class="wg-sub">嵌入表把 ' + D.tokens.length + ' 个 token 查成 shape (' + D.tokens.length +
      ', 8) 的向量；左图用前三个主成分把它压成 (' + D.tokens.length +
      ', 3) 的三维坐标。同簇 token 挨得近、异簇离得远——拖拽转一转，点一个点看看。</p>' +
      '<div class="te3-main">' +
        '<div class="te3-left">' +
          '<div class="te3-wrap">' +
            '<canvas class="wg-canvas" data-role="scene" height="350"></canvas>' +
            '<div class="te3-tip" data-role="tip" hidden></div>' +
          '</div>' +
          '<div class="te3-legend" data-role="legend"></div>' +
        '</div>' +
        '<div class="te3-side" data-role="side"></div>' +
      '</div>' +
      '<p class="wg-note">操作：拖拽 = 旋转视角 · 滚轮 = 缩放 · 点击圆点 = 选中该 token，' +
      '右侧显示它的簇名、3D 坐标与原始 8 维向量。键盘 ← → ↑ ↓ 也能旋转，Enter 换选下一个 token。</p>' +
      (D.note ? '<p class="wg-note" data-role="note"></p>' : ''));

    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };
    var scene = q('scene'), tip = q('tip'), side = q('side'), legend = q('legend');
    var N = D.tokens.length;
    scene.setAttribute('tabindex', '0');
    scene.setAttribute('aria-label', N + ' 个 token 按簇着色的三维散点图：可拖拽旋转、滚轮缩放、点击选中一个 token');
    if (D.note) { q('note').textContent = D.note; }

    // 等比归一化到半径 RADIUS 的球内（保持主成分比例）；FLOOR 为地面网格高度。
    var RADIUS = 2.05, FLOOR = -2.02, AXIS = 2.3, PERSP = 7.5;
    var maxR = 0;
    D.xyz.forEach(function (p) { maxR = Math.max(maxR, Math.hypot(p[0], p[1], p[2])); });
    var k = maxR > 0 ? RADIUS / maxR : 1;
    var pts = D.xyz.map(function (p) { return [p[0] * k, p[1] * k, p[2] * k]; });

    var view = { yaw: -0.58, pitch: 0.33, zoom: 1 };
    var sel = 0, hover = -1, raf = 0;
    var projected = [];

    var legendHtml = '';
    D.uniq.forEach(function (nm, ci) {
      var cnt = D.names.reduce(function (s, x) { return s + (x === nm ? 1 : 0); }, 0);
      legendHtml += '<span class="te3-key"><i style="background:' + PALETTE[ci % PALETTE.length] +
        '"></i>' + nm + ' ×' + cnt + '</span>';
    });
    legend.innerHTML = legendHtml;

    /* HiDPI：显式钉 style 宽高，再用 devicePixelRatio 放大位图并重置变换。 */
    function sizeCanvas(canvas, logicalHeight, fallbackMax) {
      var cssWidth = canvas.parentElement ? canvas.parentElement.clientWidth : 0;
      cssWidth = Math.max(140, Math.min(cssWidth || 400, fallbackMax || 640)) - 2;
      var ratio = window.devicePixelRatio || 1;
      canvas.style.width = cssWidth + 'px';
      canvas.style.height = logicalHeight + 'px';
      canvas.width = Math.round(cssWidth * ratio);
      canvas.height = Math.round(logicalHeight * ratio);
      var ctx = canvas.getContext('2d');
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return { ctx: ctx, w: cssWidth, h: logicalHeight };
    }

    /* 固定轻透视：yaw 绕 Y、pitch 绕屏幕 X，z2 越大离观察者越近。 */
    function geom(c) {
      var cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
      var cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
      var unit = Math.min(c.w, c.h) * 0.19 * view.zoom;
      return function (p) {
        var x1 = p[0] * cy + p[2] * sy;
        var z1 = -p[0] * sy + p[2] * cy;
        var y1 = p[1];
        var y2 = y1 * cp - z1 * sp;
        var z2 = y1 * sp + z1 * cp;
        var s = PERSP / (PERSP - z2);
        return { x: c.w / 2 + x1 * s * unit, y: c.h / 2 - y2 * s * unit, s: s, z: z2 };
      };
    }

    function drawScene() {
      var c = sizeCanvas(scene, 350, 640), ctx = c.ctx, P = geom(c);
      ctx.clearRect(0, 0, c.w, c.h);

      // 地面网格：给透视一个深度参照。
      ctx.strokeStyle = 'rgba(7,19,33,.09)'; ctx.lineWidth = 1;
      for (var g = -2; g <= 2; g++) {
        var a = P([g, FLOOR, -2]), b = P([g, FLOOR, 2]);
        var d = P([-2, FLOOR, g]), e = P([2, FLOOR, g]);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(d.x, d.y); ctx.lineTo(e.x, e.y); ctx.stroke();
      }

      // 三条坐标轴：正半轴实线、负半轴虚线，端点标 X/Y/Z。
      var o = P([0, 0, 0]);
      [[AXIS, 0, 0, 'X'], [0, AXIS, 0, 'Y'], [0, 0, AXIS, 'Z']].forEach(function (ax) {
        var pos = P([ax[0], ax[1], ax[2]]), neg = P([-ax[0], -ax[1], -ax[2]]);
        ctx.strokeStyle = 'rgba(7,19,33,.22)'; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(neg.x, neg.y); ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = 'rgba(7,19,33,.7)'; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(pos.x, pos.y); ctx.stroke();
        ctx.lineWidth = 1;
        var nx = pos.x - o.x, ny = pos.y - o.y, nl = Math.hypot(nx, ny) || 1;
        ctx.fillStyle = INKSOFT; ctx.font = '700 10px ' + MONO;
        ctx.fillText(ax[3], pos.x + nx / nl * 11 - 3, pos.y + ny / nl * 11 + 3.5);
      });

      // 远处先画；选中点最后画，保证在最上层。
      projected = pts.map(function (p, i) {
        var pr = P(p);
        pr.i = i;
        pr.r = 5 * pr.s;
        return pr;
      });
      var order = projected.slice().sort(function (a, b) { return a.z - b.z; });
      order.forEach(function (pr) {
        if (pr.i === sel) { return; }
        if (pr.i === hover) {
          ctx.strokeStyle = INK; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(pr.x, pr.y, pr.r + 2.5, 0, Math.PI * 2); ctx.stroke();
          ctx.lineWidth = 1;
        }
        ctx.fillStyle = PALETTE[D.indexOf[D.names[pr.i]] % PALETTE.length];
        ctx.beginPath(); ctx.arc(pr.x, pr.y, pr.r, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = INK; ctx.lineWidth = 1.2; ctx.stroke(); ctx.lineWidth = 1;
      });

      // 选中点：地面垂线 + 荧光绿环。
      var sp = projected[sel];
      if (sp) {
        var fp = P([pts[sel][0], FLOOR, pts[sel][2]]);
        ctx.strokeStyle = 'rgba(7,19,33,.5)'; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(sp.x, sp.y); ctx.lineTo(fp.x, fp.y); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = INK; ctx.fillRect(fp.x - 2.5, fp.y - 2.5, 5, 5);
        var r = sp.r * 1.2 + 2;
        ctx.strokeStyle = NEON; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(sp.x, sp.y, r + 2.5, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = PALETTE[D.indexOf[D.names[sel]] % PALETTE.length];
        ctx.beginPath(); ctx.arc(sp.x, sp.y, r, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = INK; ctx.lineWidth = 1.5; ctx.stroke(); ctx.lineWidth = 1;
      }

      // 角落读数与操作提示。
      var TOP_TXT = 'PCA · ' + N + ' TOKEN · ' + D.uniq.length + ' 簇';
      var BOT_TXT = '拖拽旋转 · 滚轮缩放 · 点击选中';
      ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
      ctx.fillStyle = 'rgba(7,19,33,.5)'; ctx.font = '700 9px ' + MONO;
      ctx.fillText(TOP_TXT, 9, 15);
      ctx.fillText(BOT_TXT, 9, c.h - 9);

      // 标签避让：先沿「点 → 本簇屏幕重心」的放射方向放，再试固定候选；
      // 压到别的点 / 已放标签 / 出界都扣分，取总分最低的候选。
      var centroid = {}, ccount = {};
      projected.forEach(function (pr) {
        var nm = D.names[pr.i];
        if (!centroid[nm]) { centroid[nm] = [0, 0]; ccount[nm] = 0; }
        centroid[nm][0] += pr.x; centroid[nm][1] += pr.y;
        ccount[nm] += 1;
      });
      D.uniq.forEach(function (nm) {
        centroid[nm][0] /= ccount[nm]; centroid[nm][1] /= ccount[nm];
      });
      ctx.font = '700 9px ' + MONO;
      var boxes = [
        { x0: 7, y0: 5, x1: 11 + ctx.measureText(TOP_TXT).width, y1: 17 },
        { x0: 7, y0: c.h - 17, x1: 11 + ctx.measureText(BOT_TXT).width, y1: c.h - 5 },
      ];

      function hitRect(a, b) { return a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1; }
      function hitDot(b, p, rr) {
        var nx = Math.max(b.x0, Math.min(p.x, b.x1));
        var ny = Math.max(b.y0, Math.min(p.y, b.y1));
        var dx = p.x - nx, dy = p.y - ny;
        return dx * dx + dy * dy < rr * rr;
      }
      function place(pr, fontPx, weight) {
        var font = weight + ' ' + fontPx + 'px ' + MONO;
        ctx.font = font;
        var tw = ctx.measureText(D.tokens[pr.i]).width + 2, th = fontPx + 4;
        var cx = centroid[D.names[pr.i]];
        var cands = [];
        var rx = pr.x - cx[0], ry = pr.y - cx[1], rl = Math.hypot(rx, ry);
        if (rl > 6) {
          cands.push([rx / rl, ry / rl, Math.abs(rx) >= Math.abs(ry) ? (rx > 0 ? 'left' : 'right') : 'center']);
        }
        cands = cands.concat([
          [1, 0, 'left'], [0, -1, 'center'], [0, 1, 'center'], [-1, 0, 'right'],
          [0.72, -0.72, 'left'], [0.72, 0.72, 'left'], [-0.72, -0.72, 'right'], [-0.72, 0.72, 'right'],
        ]);
        var best = null, bestScore = 1e9;
        for (var ci = 0; ci < cands.length; ci++) {
          var cd = cands[ci];
          var d = pr.r + (cd[2] === 'center' ? th / 2 + 2 : 5);
          var px = pr.x + cd[0] * d, py = pr.y + cd[1] * d;
          var x0 = cd[2] === 'right' ? px - tw : (cd[2] === 'center' ? px - tw / 2 : px);
          var b = { x0: x0, y0: py - th / 2, x1: x0 + tw, y1: py + th / 2 };
          var score = ci * 0.01;
          if (b.x0 < 2 || b.x1 > c.w - 2 || b.y0 < 2 || b.y1 > c.h - 2) { score += 100; }
          for (var bi = 0; bi < boxes.length; bi++) { if (hitRect(b, boxes[bi])) { score += 10; } }
          for (var pi = 0; pi < projected.length; pi++) {
            var q = projected[pi];
            if (q.i !== pr.i && hitDot(b, q, q.r + 2)) { score += 3; }
          }
          if (score < bestScore) { bestScore = score; best = { x: px, y: py, align: cd[2], font: font, box: b }; }
        }
        boxes.push(best.box);
        return best;
      }

      // 近处的标签先占位（视觉上层优先），选中项最优先；绘制仍按远 → 近。
      var nearFirst = projected.slice().sort(function (a, b) { return b.z - a.z; });
      var placements = {};
      if (sp) { placements[sel] = place(sp, 13, '800'); }
      nearFirst.forEach(function (pr) {
        if (pr.i === sel) { return; }
        placements[pr.i] = place(pr, 10, '700');
      });
      order.forEach(function (pr) {
        if (pr.i === sel) { return; }
        var pl = placements[pr.i];
        label(ctx, D.tokens[pr.i], pl.x, pl.y, pl.font, INKSOFT, pl.align);
      });
      if (sp) {
        label(ctx, D.tokens[sel], placements[sel].x, placements[sel].y,
          placements[sel].font, INK, placements[sel].align);
      }
      ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
    }

    /* 标签先描纸色边再填充，压在网格线上也清晰；垂直居中对齐锚点。 */
    function label(ctx, text, x, y, font, color, align) {
      ctx.font = font;
      ctx.textAlign = align || 'left';
      ctx.textBaseline = 'middle';
      ctx.strokeStyle = PAPER; ctx.lineWidth = 3; ctx.strokeText(text, x, y);
      ctx.fillStyle = color; ctx.fillText(text, x, y);
      ctx.lineWidth = 1;
    }

    function fmt(v) { return (v >= 0 ? '+' : '-') + Math.abs(v).toFixed(2); }

    /* 8 维原始向量条形图：正值蓝、负值红；只给幅值最大的那维标数值。 */
    function drawBars(canvas, values) {
      var c = sizeCanvas(canvas, 116, 240), ctx = c.ctx;
      ctx.clearRect(0, 0, c.w, c.h);
      var maxAbs = 0, maxI = 0;
      values.forEach(function (v, i) {
        if (Math.abs(v) > maxAbs) { maxAbs = Math.abs(v); maxI = i; }
      });
      if (maxAbs < 1e-9) { maxAbs = 1; }
      var mid = 46, half = 30, slot = c.w / 8, bw = Math.min(15, slot - 7);
      ctx.strokeStyle = 'rgba(7,19,33,.55)';
      ctx.beginPath(); ctx.moveTo(2, mid); ctx.lineTo(c.w - 2, mid); ctx.stroke();
      ctx.font = '700 8.5px ' + MONO;
      for (var i = 0; i < 8; i++) {
        var v = values[i], len = Math.abs(v) / maxAbs * half;
        var x = slot * i + (slot - bw) / 2;
        var y = v >= 0 ? mid - len : mid;
        var h = Math.max(1, len);
        ctx.fillStyle = v >= 0 ? BLUE : RED;
        var radii = v >= 0 ? [2, 2, 0, 0] : [0, 0, 2, 2];
        if (ctx.roundRect) {
          ctx.beginPath(); ctx.roundRect(x, y, bw, h, radii); ctx.fill();
        } else {
          ctx.fillRect(x, y, bw, h);
        }
        ctx.fillStyle = MUTED;
        ctx.fillText('d' + i, slot * i + slot / 2 - 6, c.h - 6);
        if (i === maxI) {
          ctx.fillStyle = v >= 0 ? '#3a4ba0' : '#9c3a30';
          ctx.fillText((v >= 0 ? '+' : '-') + Math.abs(v).toFixed(2),
            slot * i + slot / 2 - 12, v >= 0 ? y - 4 : y + h + 9);
        }
      }
      return maxAbs;
    }

    function renderSide() {
      if (sel < 0) {
        side.innerHTML =
          '<div class="te3-empty">未选中：点击图中任一圆点，查看该 token 的簇名、3D 坐标与原始 8 维向量。</div>';
        return;
      }
      var nm = D.names[sel];
      var col = PALETTE[D.indexOf[nm] % PALETTE.length];
      side.innerHTML =
        '<div class="wg-label"><span>选中的 token</span><span>№ ' + (sel + 1) + '/' + N + '</span></div>' +
        '<div class="te3-tok">' +
          '<span class="te3-glyph" style="background:' + col + '">' + D.tokens[sel] + '</span>' +
          '<div class="te3-meta"><b>「' + D.tokens[sel] + '」</b><span>簇 · ' + nm + '</span></div>' +
        '</div>' +
        '<div class="wg-label"><span>原始 8 维向量</span><span data-role="maxv"></span></div>' +
        '<div class="te3-barswrap"><canvas class="wg-canvas" data-role="bars" height="116"></canvas></div>' +
        '<div class="wg-label"><span>3D 坐标 (PCA)</span><span>半径 ' +
          Math.hypot(D.xyz[sel][0], D.xyz[sel][1], D.xyz[sel][2]).toFixed(2) + '</span></div>' +
        '<div class="te3-coord">' +
          '<div><b>' + fmt(D.xyz[sel][0]) + '</b><span>x</span></div>' +
          '<div><b>' + fmt(D.xyz[sel][1]) + '</b><span>y</span></div>' +
          '<div><b>' + fmt(D.xyz[sel][2]) + '</b><span>z</span></div>' +
        '</div>';
      var maxAbs = drawBars(side.querySelector('[data-role="bars"]'), D.vec8[sel]);
      side.querySelector('[data-role="maxv"]').textContent = 'max|v|=' + maxAbs.toFixed(2);
    }

    function requestDraw() {
      if (raf) { return; }
      raf = window.requestAnimationFrame(function () { raf = 0; drawScene(); });
    }

    /* 命中判定半径大于圆点本身，方便点选。 */
    function nearest(mx, my, limit) {
      var best = -1, bd = limit * limit;
      projected.forEach(function (p) {
        var dx = p.x - mx, dy = p.y - my, d = dx * dx + dy * dy;
        if (d <= bd) { bd = d; best = p.i; }
      });
      return best;
    }

    function localXY(e) {
      var r = scene.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    }

    var drag = null;
    scene.addEventListener('pointerdown', function (e) {
      if (e.button !== undefined && e.button !== 0) { return; }
      drag = { x: e.clientX, y: e.clientY, moved: 0, id: e.pointerId };
      try { scene.setPointerCapture(e.pointerId); } catch (err) { /* 拖拽仍可用 */ }
      scene.style.cursor = 'grabbing';
      tip.hidden = true;
    });
    scene.addEventListener('pointermove', function (e) {
      if (drag) {
        var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        drag.moved += Math.abs(dx) + Math.abs(dy);
        drag.x = e.clientX; drag.y = e.clientY;
        view.yaw += dx * 0.0075;
        view.pitch = Math.max(-1.35, Math.min(1.35, view.pitch + dy * 0.0065));
        requestDraw();
        return;
      }
      var m = localXY(e);
      hover = nearest(m[0], m[1], 15);
      if (hover >= 0) {
        scene.style.cursor = 'pointer';
        var p = projected[hover];
        tip.textContent = '「' + D.tokens[hover] + '」· ' + D.names[hover];
        tip.hidden = false;
        var wrap = scene.parentElement.getBoundingClientRect();
        tip.style.left = Math.min(Math.max(p.x + 12, 0), wrap.width - tip.offsetWidth - 2) + 'px';
        tip.style.top = Math.max(p.y - 30, 0) + 'px';
      } else {
        scene.style.cursor = 'grab';
        tip.hidden = true;
      }
    });
    function endDrag(e) {
      if (!drag) { return; }
      var wasClick = drag.moved < 6;
      try { scene.releasePointerCapture(drag.id); } catch (err) { /* 已释放 */ }
      drag = null;
      scene.style.cursor = 'grab';
      if (wasClick) {
        var m = localXY(e);
        var hit = nearest(m[0], m[1], 16);
        if (hit >= 0) { sel = hit; renderSide(); requestDraw(); }
      }
    }
    scene.addEventListener('pointerup', endDrag);
    scene.addEventListener('pointercancel', function () { drag = null; scene.style.cursor = 'grab'; });
    scene.addEventListener('pointerleave', function () { hover = -1; tip.hidden = true; requestDraw(); });
    scene.addEventListener('wheel', function (e) {
      e.preventDefault();
      view.zoom = Math.max(0.55, Math.min(2.3, view.zoom * (e.deltaY < 0 ? 1.09 : 0.917)));
      requestDraw();
    }, { passive: false });
    scene.addEventListener('keydown', function (e) {
      var step = e.shiftKey ? 0.02 : 0.1;
      var used = true;
      if (e.key === 'ArrowLeft') { view.yaw -= step; }
      else if (e.key === 'ArrowRight') { view.yaw += step; }
      else if (e.key === 'ArrowUp') { view.pitch = Math.min(1.35, view.pitch + step); }
      else if (e.key === 'ArrowDown') { view.pitch = Math.max(-1.35, view.pitch - step); }
      else if (e.key === 'Enter') { sel = (sel + 1) % N; renderSide(); }
      else { used = false; }
      if (used) { e.preventDefault(); requestDraw(); }
    });

    var onResize = function () {
      requestDraw();
      renderSide();
    };
    window.addEventListener('resize', onResize);

    renderSide();
    drawScene();

    return function dispose() {
      if (raf) { window.cancelAnimationFrame(raf); raf = 0; }
      window.removeEventListener('resize', onResize);
    };
  }

  window.DLWidget.register('token-embed-3d', mount);
})();
