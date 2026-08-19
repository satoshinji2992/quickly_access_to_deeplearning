/* image-batch — 8 张真实 CIFAR-100 图片：点击选中一张，看 R/G/B 三张
 * 通道热图，再点一个像素，读出它在 NCHW 布局里的四个下标 (B, C, H, W)。
 * 数据资产：assets/widgets/cifar8.png（256×32 横向 sprite）+ cifar8_meta.json。 */
(function () {
  'use strict';

  var BASE = '/quickly_access_to_deeplearning/assets/widgets/';
  var SPRITE_URL = BASE + 'cifar8.png';
  var META_URL = BASE + 'cifar8_meta.json';
  /* meta 拉取失败时的兜底：sprite 的 8 张图是固定挑选并按此顺序生成的。 */
  var FALLBACK_META = [
    ['cattle', '牛'], ['dinosaur', '恐龙'], ['apple', '苹果'], ['boy', '男孩'],
    ['aquarium_fish', '鱼'], ['telephone', '电话'], ['train', '火车'], ['cup', '杯子'],
  ];

  var N = 8, SIDE = 32, SPRITE_W = N * SIDE;
  var INK = '#071321', BLUE = '#0b63f3', NEON = '#c8ff47';
  var MONO = 'ui-monospace, SFMono-Regular, Consolas, monospace';
  var CH_HEX = ['#d0342c', '#15803d', '#2456e0'];
  var CH_ZH = ['R', 'G', 'B'];

  var STYLE_TEXT = [
    '/* 缩略图行：batch 里的 8 张图，B=0..7 */',
    '.ib-books{--bs:56px;display:flex;justify-content:center;gap:13px;align-items:flex-end;padding:6px 6px 0}',
    '.ib-book{width:var(--bs);height:var(--bs);border:1px solid #071321;background:#f7f5ef;',
    'background-repeat:no-repeat;image-rendering:pixelated;',
    'background-size:calc(var(--bs)*8) var(--bs);',
    'cursor:pointer;',
    'transition:transform .16s ease,box-shadow .16s ease,opacity .16s ease}',
    '.ib-book:hover,.ib-book:focus-visible{transform:translateY(-4px);',
    'box-shadow:3px 3px 0 #0b63f3;outline:none}',
    '.ib-book.is-on{transform:translateY(-4px);box-shadow:2px 2px 0 #0b63f3}',
    '.ib-book.is-dim{opacity:.3}',
    '.ib-names{display:flex;justify-content:center;gap:13px;margin-top:6px}',
    '.ib-name{width:var(--bs);text-align:center;line-height:1.4}',
    '.ib-name b{display:block;font:700 11px ' + MONO + ';color:#263541;white-space:nowrap}',
    '.ib-name span{font:700 9px ' + MONO + ';color:' + BLUE + '}',
    '.ib-name.is-dim{opacity:.38}',
    '/* 展开面板 */',
    '.ib-panel{margin-top:14px;border:1px solid rgba(7,19,33,.3);background:#f4f2ec;padding:12px}',
    '.ib-ehead{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:8px}',
    '.ib-etitle{font:700 11px ' + MONO + ';letter-spacing:.05em;color:#071321}',
    '.ib-views{display:grid;grid-template-columns:1.16fr 1fr 1fr 1fr;gap:9px}',
    '.ib-view{border:1px solid rgba(7,19,33,.4);background:#fffef9;padding:6px}',
    '.ib-view.is-on{border-color:' + BLUE + ';box-shadow:2px 2px 0 ' + BLUE + '}',
    '.ib-vhead{display:flex;justify-content:space-between;gap:6px;margin-bottom:5px;',
    'font:700 9.5px ' + MONO + ';letter-spacing:.04em;color:#45515c;white-space:nowrap}',
    '.ib-vhead b{color:#071321}',
    '.ib-cwrap{line-height:0}',
    '.ib-view canvas{display:block;width:100%;image-rendering:pixelated;cursor:crosshair;',
    'touch-action:manipulation}',
    '.ib-cw{display:inline-block;width:9px;height:9px;border:1px solid #071321;margin-right:5px;vertical-align:-1px}',
    '/* NCHW 读数 */',
    '.ib-sw{display:inline-block;width:13px;height:13px;border:1px solid #071321;',
    'margin-right:8px;vertical-align:-1px;background:#fffef9}',
    '.ib-nchw{display:flex;gap:10px;align-items:stretch;margin:10px 0 0}',
    '.ib-code{flex:1 1 auto;min-width:0;border:1px solid #071321;background:#071321;',
    'padding:10px 13px;font:700 12.5px ' + MONO + ';color:#fffef9;line-height:1.85;',
    'overflow-x:auto;white-space:nowrap}',
    '.ib-code em{font-style:normal;color:' + NEON + '}',
    '.ib-code .ib-dim{color:rgba(255,254,249,.72)}',
    '.ib-code .ib-cap{font:600 10px ' + MONO + ';color:rgba(255,254,249,.55);margin-top:3px;',
    'white-space:normal}',
    '.ib-side{flex:0 0 168px;border:1px solid rgba(7,19,33,.3);background:#fffef9;padding:9px}',
    '.ib-side .wg-label{margin:0 0 6px}',
    '.ib-chips{display:grid;grid-template-columns:1fr 1fr;gap:5px}',
    '.ib-chip{display:flex;align-items:center;justify-content:center;border:1px solid #071321;',
    'background:#fffef9;padding:6px 4px;font:700 10px ' + MONO + ';color:#071321;cursor:pointer}',
    '.ib-chip:hover{box-shadow:2px 2px 0 ' + BLUE + '}',
    '.ib-chip.is-on{background:#071321;color:#fffef9}',
    '@media (max-width:620px){',
    '.ib-books{gap:6px}.ib-names{gap:6px}.ib-book,.ib-name{--bs:38px}',
    '.ib-views{grid-template-columns:1fr 1fr}',
    '.ib-nchw{flex-direction:column}.ib-side{flex:auto}',
    '}',
  ].join('\n');

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  /* 解析 meta：8 项、index 0..7、有 zh/label 才算合法，否则用兜底。 */
  function parseMeta(raw) {
    var out = null;
    if (raw && Array.isArray(raw.images) && raw.images.length === N) {
      var ok = true, tmp = [];
      for (var i = 0; i < N; i++) {
        var it = raw.images[i];
        if (!it || it.index !== i || !it.zh) { ok = false; break; }
        tmp.push({ zh: String(it.zh), label: it.label ? String(it.label) : '' });
      }
      if (ok) { out = tmp; }
    }
    if (!out) {
      out = FALLBACK_META.map(function (p, i) { return { zh: p[1], label: p[0] }; });
    }
    return out;
  }

  function mount(container) {
    var styleEl = document.createElement('style');
    styleEl.textContent = STYLE_TEXT;
    var disposed = false;
    var innerDispose = null;

    /* 私有样式随内容一起重建（加载态/错误态/主体态都用 .ib-* 类）。 */
    function setBody(html) {
      container.innerHTML = html;
      container.appendChild(styleEl);
    }

    function showError(msg) {
      setBody(
        '<p class="wg-title">一个 batch，8 张 CIFAR 图片</p>' +
        '<p class="wg-note" style="border:1px solid #c0392b;background:rgba(192,57,43,.07);' +
        'color:#c0392b;padding:10px 12px;font-family:' + MONO + '">图片数据加载失败：' + esc(msg) +
        '<br>数据文件应位于 <code>' + SPRITE_URL + '</code>。请确认文件存在后刷新页面。</p>');
    }

    setBody(
      '<p class="wg-title">一个 batch，8 张 CIFAR 图片</p>' +
      '<p class="wg-sub">正在从本地加载 CIFAR-100 图片 sprite…</p>');

    var metaP = fetch(META_URL)
      .then(function (r) { if (!r.ok) { throw new Error('HTTP ' + r.status); } return r.json(); })
      .catch(function () { return null; });   /* meta 可兜底，不致命 */

    var spriteP = new Promise(function (resolve, reject) {
      var img = new Image();
      img.onload = function () {
        if (img.naturalWidth !== SPRITE_W || img.naturalHeight !== SIDE) {
          reject(new Error('sprite 尺寸应为 ' + SPRITE_W + '×' + SIDE +
            '，实际 ' + img.naturalWidth + '×' + img.naturalHeight));
          return;
        }
        resolve(img);
      };
      img.onerror = function () { reject(new Error('图片无法解码')); };
      img.src = SPRITE_URL;
    });

    spriteP.then(function (img) {
      return metaP.then(function (rawMeta) { return { img: img, meta: parseMeta(rawMeta) }; });
    }).then(function (assets) {
      if (disposed) { return; }
      innerDispose = build(container, assets.meta, assets.img);
    }).catch(function (e) {
      if (!disposed) { showError(e && e.message ? e.message : String(e)); }
    });

    return function dispose() {
      disposed = true;
      if (innerDispose) { innerDispose(); innerDispose = null; }
    };
  }

  function build(container, META, IMG) {
    /* mount 阶段已把私有 <style> 挂进 container；这里持有引用，随内容重建。 */
    var styleElShared = container.querySelector('style');
    function setBody(html) {
      container.innerHTML = html;
      container.appendChild(styleElShared);
    }
    setBody(
      '<p class="wg-title">一个 batch，8 张 CIFAR 图片</p>' +
      '<p class="wg-sub">每张 32×32、3 个通道，摞成一个 shape 为 (8,&nbsp;3,&nbsp;32,&nbsp;32) 的数组 X，' +
      '四个轴依次是 (B, C, H, W)。点击一张图片选中它，再点放大图上的一个像素，读出它的四个下标。</p>' +
      '<div class="ib-books" data-role="books"></div>' +
      '<div class="ib-names" data-role="names"></div>' +
      '<p class="wg-note" data-role="hint" style="text-align:center"></p>' +
      '<div class="ib-panel" data-role="panel">' +
        '<div class="ib-ehead">' +
          '<div class="ib-etitle" data-role="etitle"></div>' +
          '<button type="button" class="wg-button" data-role="collapse">收起</button>' +
        '</div>' +
        '<div class="ib-views">' +
          '<div class="ib-view" data-role="view-all">' +
            '<div class="ib-vhead"><span>原图 · RGB</span><b>32×32</b></div>' +
            '<div class="ib-cwrap"><canvas data-role="cv-all" aria-label="选中的 CIFAR 图片放大图，移动或点击可选择像素"></canvas></div>' +
          '</div>' +
          '<div class="ib-view" data-role="view-0">' +
            '<div class="ib-vhead"><span><i class="ib-cw" style="background:' + CH_HEX[0] + '"></i>C=0 · R</span><b data-role="hv-0"></b></div>' +
            '<div class="ib-cwrap"><canvas data-role="cv-0" aria-label="红色通道热图，移动或点击可选择像素"></canvas></div>' +
          '</div>' +
          '<div class="ib-view" data-role="view-1">' +
            '<div class="ib-vhead"><span><i class="ib-cw" style="background:' + CH_HEX[1] + '"></i>C=1 · G</span><b data-role="hv-1"></b></div>' +
            '<div class="ib-cwrap"><canvas data-role="cv-1" aria-label="绿色通道热图，移动或点击可选择像素"></canvas></div>' +
          '</div>' +
          '<div class="ib-view" data-role="view-2">' +
            '<div class="ib-vhead"><span><i class="ib-cw" style="background:' + CH_HEX[2] + '"></i>C=2 · B</span><b data-role="hv-2"></b></div>' +
            '<div class="ib-cwrap"><canvas data-role="cv-2" aria-label="蓝色通道热图，移动或点击可选择像素"></canvas></div>' +
          '</div>' +
        '</div>' +
        '<div class="wg-label" style="margin-top:12px"><span>点一个像素</span><span>在任一视图上移动或点击</span></div>' +
        '<div class="wg-readout" style="margin-top:4px">' +
          '<div class="wg-stat"><span>像素坐标 (h, w)</span><b data-role="st-hw"></b></div>' +
          '<div class="wg-stat"><span>R 值 · C=0</span><b data-role="st-0"></b></div>' +
          '<div class="wg-stat"><span>G 值 · C=1</span><b data-role="st-1"></b></div>' +
          '<div class="wg-stat"><span>B 值 · C=2</span><b data-role="st-2"></b></div>' +
        '</div>' +
        '<div class="ib-nchw">' +
          '<div class="ib-code">' +
            '<div data-role="line1"></div>' +
            '<div class="ib-dim" data-role="line2"></div>' +
            '<div class="ib-cap" data-role="cap"></div>' +
          '</div>' +
          '<div class="ib-side">' +
            '<div class="wg-label"><span>正在看的通道</span></div>' +
            '<div class="ib-chips" data-role="chips"></div>' +
          '</div>' +
        '</div>' +
      '</div>' +
      '<p class="wg-note">每张缩略图 = batch 里的一张图（B = 0..7）；三张热图把 R/G/B 拆开，' +
      '颜色越亮该通道数值越大；冒号 : 表示这一维整条取。batch 里每张图内容不同，但 shape 必须完全一致，才能拼进同一个数组。</p>');

    var q = function (role) { return container.querySelector('[data-role="' + role + '"]'); };
    var S = {
      sel: 0,            /* 选中的图片 = batch 下标 B */
      expanded: true,
      h: 12, w: 20,      /* 默认选中的像素 */
      activeC: 0,        /* X[b, c, h, w] 里高亮的 c */
      activeView: 0,     /* 'all' | 0 | 1 | 2：正在看哪个视图 */
    };

    /* ---- 像素数据：sprite → RGBA ---- */
    var off = document.createElement('canvas');
    off.width = SPRITE_W; off.height = SIDE;
    var octx = off.getContext('2d', { willReadFrequently: true });
    octx.drawImage(IMG, 0, 0);
    var PIXELS = octx.getImageData(0, 0, SPRITE_W, SIDE).data;
    function px(b, h, w, c) { return PIXELS[((h * SPRITE_W) + (b * SIDE + w)) * 4 + c]; }

    /* 单通道热图离屏缓存：只在换图时重建。 */
    var chan = { sel: -1, cvs: [null, null, null] };
    function chanCanvas(c) {
      if (chan.sel !== S.sel) {
        chan.cvs = [0, 1, 2].map(function () {
          var cv = document.createElement('canvas');
          cv.width = SIDE; cv.height = SIDE;
          return cv;
        });
        for (var k = 0; k < 3; k++) {
          var ctx = chan.cvs[k].getContext('2d');
          var imgd = ctx.createImageData(SIDE, SIDE);
          for (var h = 0; h < SIDE; h++) {
            for (var w = 0; w < SIDE; w++) {
              var i = (h * SIDE + w) * 4, v = px(S.sel, h, w, k);
              imgd.data[i + k] = v;
              imgd.data[i + 3] = 255;
            }
          }
          ctx.putImageData(imgd, 0, 0);
        }
        chan.sel = S.sel;
      }
      return chan.cvs[c];
    }

    /* ---- 缩略图行 DOM ---- */
    var booksEl = q('books'), namesEl = q('names');
    var bookEls = [], nameEls = [];
    META.forEach(function (m, i) {
      var b = document.createElement('div');
      b.className = 'ib-book';
      b.setAttribute('role', 'button');
      b.setAttribute('tabindex', '0');
      b.setAttribute('aria-label', '选择第 ' + (i + 1) + ' 张图：' + m.zh);
      b.style.backgroundImage = 'url(' + SPRITE_URL + ')';
      b.style.backgroundPosition = 'calc(var(--bs) * ' + (-i) + ') 0';
      b.addEventListener('click', function () { pickBook(i); });
      b.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pickBook(i); }
      });
      booksEl.appendChild(b);
      bookEls.push(b);

      var n = document.createElement('div');
      n.className = 'ib-name';
      n.innerHTML = '<b>' + esc(m.zh) + '</b><span>B=' + i + '</span>';
      namesEl.appendChild(n);
      nameEls.push(n);
    });

    function pickBook(i) {
      if (S.expanded && S.sel === i) { S.expanded = false; }   /* 再点一次 = 收起 */
      else { S.sel = i; S.expanded = true; }
      render();
    }

    q('collapse').addEventListener('click', function () { S.expanded = false; render(); });

    /* ---- 通道选择 chips（4 格通道示意） ---- */
    var chipsEl = q('chips');
    var chipEls = {};
    [['all', '原图 · :'], ['0', null], ['1', null], ['2', null]].forEach(function (pair) {
      var key = pair[0], btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'ib-chip';
      if (key === 'all') {
        btn.innerHTML = pair[1];
      } else {
        var c = parseInt(key, 10);
        btn.innerHTML = '<i class="ib-cw" style="background:' + CH_HEX[c] + '"></i>' +
          CH_ZH[c] + ' · C=' + key;
      }
      btn.addEventListener('click', function () {
        S.activeView = key === 'all' ? 'all' : parseInt(key, 10);
        if (key !== 'all') { S.activeC = S.activeView; }
        render();
      });
      chipsEl.appendChild(btn);
      chipEls[key] = btn;
    });

    /* ---- 视图 canvas 与取像素 ---- */
    var views = [
      { key: 'all', cv: q('cv-all'), wrap: q('view-all') },
      { key: 0, cv: q('cv-0'), wrap: q('view-0') },
      { key: 1, cv: q('cv-1'), wrap: q('view-1') },
      { key: 2, cv: q('cv-2'), wrap: q('view-2') },
    ];

    /* 显式钉 canvas 的 CSS 尺寸（HiDPI 下避免画布撑爆布局），
     * 并把 backing store 设为 css × dpr。 */
    function fit(canvas) {
      var w = Math.round(canvas.parentElement.clientWidth);
      if (!(w > 8)) { return null; }
      var ratio = window.devicePixelRatio || 1;
      canvas.style.width = w + 'px';
      canvas.style.height = w + 'px';
      canvas.width = Math.round(w * ratio);
      canvas.height = Math.round(w * ratio);
      var ctx = canvas.getContext('2d');
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return { ctx: ctx, w: w };
    }

    function paint(view) {
      var f = fit(view.cv);
      if (!f) { return; }
      var ctx = f.ctx, W = f.w;
      ctx.clearRect(0, 0, W, W);
      ctx.imageSmoothingEnabled = false;
      if (view.key === 'all') {
        ctx.drawImage(IMG, S.sel * SIDE, 0, SIDE, SIDE, 0, 0, W, W);
      } else {
        ctx.drawImage(chanCanvas(view.key), 0, 0, W, W);
      }
      /* 十字参考线 + 选中像素的格子框 */
      var cell = W / SIDE;
      var cx = (S.w + 0.5) * cell, cy = (S.h + 0.5) * cell;
      ctx.strokeStyle = 'rgba(11,99,243,.55)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, 0); ctx.lineTo(cx, W);
      ctx.moveTo(0, cy); ctx.lineTo(W, cy);
      ctx.stroke();
      ctx.strokeStyle = NEON;
      ctx.lineWidth = 3;
      ctx.strokeRect(S.w * cell + 1.5, S.h * cell + 1.5, cell - 3, cell - 3);
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1;
      ctx.strokeRect(S.w * cell + 0.5, S.h * cell + 0.5, cell - 1, cell - 1);
    }

    function pickPixel(view, e) {
      var r = view.cv.getBoundingClientRect();
      if (!(r.width > 0)) { return; }
      S.w = Math.max(0, Math.min(SIDE - 1, Math.floor((e.clientX - r.left) / r.width * SIDE)));
      S.h = Math.max(0, Math.min(SIDE - 1, Math.floor((e.clientY - r.top) / r.height * SIDE)));
      S.activeView = view.key;
      if (view.key !== 'all') { S.activeC = view.key; }
      scheduleRender();
    }
    views.forEach(function (view) {
      view.cv.addEventListener('pointerdown', function (e) { pickPixel(view, e); });
      view.cv.addEventListener('pointermove', function (e) {
        if (e.pointerType === 'mouse' || e.buttons) { pickPixel(view, e); }
      });
    });

    /* ---- 渲染 ---- */
    var raf = 0;
    function scheduleRender() {
      if (raf) { return; }
      raf = window.requestAnimationFrame(function () { raf = 0; render(); });
    }

    function render() {
      var m = META[S.sel];
      bookEls.forEach(function (el, i) {
        el.classList.toggle('is-dim', S.expanded && i !== S.sel);
        el.classList.toggle('is-on', S.expanded && i === S.sel);
        nameEls[i].classList.toggle('is-dim', S.expanded && i !== S.sel);
      });
      q('hint').textContent = S.expanded
        ? '已选中第 ' + (S.sel + 1) + ' 张（B=' + S.sel + '）· 点击其他图片切换；再点当前图片或按「收起」折叠'
        : '已折叠 · 点击任意一张图片展开它的三个通道';

      var panel = q('panel');
      if (!S.expanded) {
        panel.style.display = 'none';
        return;
      }
      panel.style.display = '';

      q('etitle').textContent = '第 ' + (S.sel + 1) + '/' + N + ' 张 · ' +
        m.zh + (m.label ? '（' + m.label + '）' : '') + ' · B=' + S.sel;

      views.forEach(function (view) {
        paint(view);
        view.wrap.classList.toggle('is-on', S.activeView === view.key);
        if (view.key !== 'all') { q('hv-' + view.key).textContent = px(S.sel, S.h, S.w, view.key); }
      });

      var r = px(S.sel, S.h, S.w, 0), g = px(S.sel, S.h, S.w, 1), b = px(S.sel, S.h, S.w, 2);
      q('st-hw').textContent = '(' + S.h + ', ' + S.w + ')';
      q('st-0').innerHTML = '<i class="ib-sw" style="background:rgb(' + r + ',0,0)"></i>' + r;
      q('st-1').innerHTML = '<i class="ib-sw" style="background:rgb(0,' + g + ',0)"></i>' + g;
      q('st-2').innerHTML = '<i class="ib-sw" style="background:rgb(0,0,' + b + ')"></i>' + b;

      var cv = px(S.sel, S.h, S.w, S.activeC);
      q('line1').innerHTML = 'X[' + S.sel + ', <em>' + S.activeC + '</em>, ' + S.h + ', ' + S.w +
        '] = <em>' + cv + '</em>';
      q('line2').textContent = 'X[' + S.sel + ', :, ' + S.h + ', ' + S.w + '] = [' +
        r + ', ' + g + ', ' + b + ']';
      q('cap').textContent = '当前 batch 的第 ' + (S.sel + 1) + ' 张（B=' + S.sel + '）' +
        '· C=0/1/2 依次是 R/G/B · 单独看 C=' + S.activeC + '（' + CH_ZH[S.activeC] + '）';

      ['all', '0', '1', '2'].forEach(function (key) {
        chipEls[key].classList.toggle('is-on',
          S.activeView === (key === 'all' ? 'all' : parseInt(key, 10)));
      });
    }

    var onResize = function () { scheduleRender(); };
    window.addEventListener('resize', onResize);
    render();

    return function dispose() {
      if (raf) { window.cancelAnimationFrame(raf); raf = 0; }
      window.removeEventListener('resize', onResize);
    };
  }

  window.DLWidget.register('image-batch', mount);
})();
