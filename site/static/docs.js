/* docs.js — 文档页交互：目录抽屉、代码复制、Pagefind 搜索、
 * 阅读进度条、TOC 滚动高亮、标题锚点复制、Issue 反馈入口。 */
(function () {
  'use strict';

  /* ---- 目录抽屉（移动端） ---- */
  var menu = document.querySelector('.docs-menu');
  var sidebar = document.querySelector('.docs-sidebar');
  if (menu && sidebar) {
    menu.addEventListener('click', function () {
      var open = document.body.classList.toggle('nav-open');
      menu.setAttribute('aria-expanded', String(open));
    });
    sidebar.addEventListener('click', function (event) {
      if (event.target.closest('a')) {
        document.body.classList.remove('nav-open');
        menu.setAttribute('aria-expanded', 'false');
      }
    });
  }

  /* ---- 代码块复制 ---- */
  document.querySelectorAll('.markdown-body pre').forEach(function (pre) {
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy-code';
    button.textContent = 'COPY';
    button.addEventListener('click', function () {
      navigator.clipboard.writeText(pre.innerText.replace(/^COPY\n?/, ''));
      button.textContent = 'COPIED';
      window.setTimeout(function () { button.textContent = 'COPY'; }, 1200);
    });
    pre.appendChild(button);
  });

  var article = document.querySelector('.markdown-body');

  /* ---- 阅读进度条 ---- */
  var bar = document.createElement('div');
  bar.className = 'read-progress';
  bar.setAttribute('aria-hidden', 'true');
  document.body.appendChild(bar);
  var tick = function () {
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    bar.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + '%';
  };
  window.addEventListener('scroll', tick, { passive: true });
  tick();

  /* ---- 标题锚点复制 ---- */
  if (article) {
    article.querySelectorAll('h2, h3').forEach(function (h) {
      if (!h.id) { return; }
      var link = document.createElement('button');
      link.type = 'button';
      link.className = 'anchor-copy';
      link.textContent = '⧉';
      link.title = '复制锚点链接';
      link.setAttribute('aria-label', '复制锚点链接');
      link.addEventListener('click', function () {
        var url = location.origin + location.pathname + '#' + h.id;
        navigator.clipboard.writeText(url);
        link.textContent = '✓';
        window.setTimeout(function () { link.textContent = '⧉'; }, 1200);
      });
      h.appendChild(link);
    });
  }

  /* ---- TOC 滚动高亮 ---- */
  var toc = document.querySelector('.docs-toc');
  if (toc && article) {
    var links = {};
    toc.querySelectorAll('a').forEach(function (a) {
      var id = (a.getAttribute('href') || '').slice(1);
      if (id) { links[id] = a; }
    });
    var heads = article.querySelectorAll('h2, h3');
    if ('IntersectionObserver' in window && heads.length) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) {
            Object.values(links).forEach(function (a) { a.classList.remove('active'); });
            var a = links[e.target.id];
            if (a) { a.classList.add('active'); }
          }
        });
      }, { rootMargin: '-15% 0px -70% 0px' });
      heads.forEach(function (h) { if (h.id && links[h.id]) { io.observe(h); } });
    }
  }

  /* ---- Pagefind 全文搜索 ---- */
  var searchBtn = document.querySelector('.docs-search');
  if (!searchBtn) { return; }
  var overlay = document.createElement('div');
  overlay.className = 'search-overlay';
  overlay.innerHTML =
    '<div class="search-modal" role="dialog" aria-label="搜索文档">' +
      '<input type="search" class="search-input" placeholder="搜索 25 个页面…（Esc 关闭）" aria-label="搜索文档">' +
      '<div class="search-results"></div>' +
      '<div class="search-foot">Pagefind 静态索引 · Enter 打开第一条</div>' +
    '</div>';
  document.body.appendChild(overlay);
  var input = overlay.querySelector('.search-input');
  var resultsEl = overlay.querySelector('.search-results');
  var pagefind = null;
  var debounce = 0;

  function open() {
    overlay.classList.add('is-open');
    input.value = '';
    resultsEl.textContent = '';
    input.focus();
    if (!pagefind) {
      /* 索引仅在构建后的站点存在；本地根路径部署与线上子路径两种布局都尝试 */
      var candidates = ['/pagefind/pagefind.js', '/quickly_access_to_deeplearning/pagefind/pagefind.js'];
      var load = function (i) {
        if (i >= candidates.length) {
          resultsEl.innerHTML = '<p class="search-empty">搜索索引仅在构建后的站点可用（本地 hugo server 无 /pagefind/）。</p>';
          return;
        }
        import(candidates[i])
          .then(function (mod) { pagefind = mod; pagefind.init && pagefind.init(); })
          .catch(function () { load(i + 1); });
      };
      load(0);
    }
  }
  function close() { overlay.classList.remove('is-open'); }

  searchBtn.addEventListener('click', open);
  overlay.addEventListener('click', function (e) { if (e.target === overlay) { close(); } });
  document.addEventListener('keydown', function (e) {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); overlay.classList.contains('is-open') ? close() : open(); }
    if (e.key === 'Escape') { close(); }
    if (e.key === 'Enter' && overlay.classList.contains('is-open')) {
      var first = resultsEl.querySelector('a');
      if (first) { location.href = first.getAttribute('href'); }
    }
  });

  input.addEventListener('input', function () {
    if (!pagefind) { return; }
    window.clearTimeout(debounce);
    var query = input.value.trim();
    if (!query) { resultsEl.textContent = ''; return; }
    debounce = window.setTimeout(function () {
      pagefind.search(query).then(function (res) {
        if (!res.results.length) {
          resultsEl.innerHTML = '<p class="search-empty">没有匹配的页面。</p>';
          return;
        }
        var top = res.results.slice(0, 8);
        Promise.all(top.map(function (r) { return r.data(); })).then(function (datas) {
          resultsEl.innerHTML = '';
          datas.forEach(function (d) {
            var a = document.createElement('a');
            a.href = d.url;
            a.innerHTML = '<b>' + (d.meta && d.meta.title ? d.meta.title : d.url) + '</b>' +
              '<span>' + (d.excerpt || '') + '</span>';
            resultsEl.appendChild(a);
          });
        });
      });
    }, 160);
  });
})();
