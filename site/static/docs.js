/* docs.js — 文档页交互：目录抽屉、代码复制、Pagefind 搜索、
 * 阅读进度条、TOC 滚动高亮、标题锚点复制、Issue 反馈入口。 */
(function () {
  'use strict';

  var scriptURL = document.currentScript.src;

  async function copyText(button, text, label, doneLabel) {
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = doneLabel;
    } catch (_) {
      button.textContent = '复制失败';
    }
    window.setTimeout(function () { button.textContent = label; }, 1200);
  }

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
    var code = pre.querySelector('code');
    if (!code) { return; }
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy-code';
    button.textContent = 'COPY';
    button.setAttribute('aria-label', '复制代码');
    button.setAttribute('aria-live', 'polite');
    button.addEventListener('click', function () {
      copyText(button, code.textContent, 'COPY', 'COPIED');
    });
    pre.appendChild(button);
  });

  var article = document.querySelector('.markdown-body');
  function revealHash() {
    var id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch (_) { return; }
    var target = document.getElementById(id);
    if (!target) { return; }
    var parent = target.parentElement;
    var opened = false;
    while (parent) {
      if (parent.tagName === 'DETAILS' && !parent.open) { parent.open = true; opened = true; }
      parent = parent.parentElement;
    }
    if (opened) { target.scrollIntoView({ block: 'start' }); }
  }
  window.addEventListener('hashchange', revealHash);
  revealHash();

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
        copyText(link, url, '⧉', '✓');
      });
      h.appendChild(link);
    });
  }

  /* ---- TOC 滚动高亮 ---- */
  var toc = document.querySelector('.docs-toc');
  if (toc && article) {
    var links = {};
    toc.querySelectorAll('a').forEach(function (a) {
      var id = decodeURIComponent((a.getAttribute('href') || '').slice(1));
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
    '<div class="search-modal" role="dialog" aria-modal="true" aria-label="搜索文档">' +
      '<button type="button" class="search-close" aria-label="关闭搜索">关闭 ×</button>' +
      '<input type="search" class="search-input" placeholder="搜索课程内容…（Esc 关闭）" aria-label="搜索文档">' +
      '<div class="search-results" aria-live="polite"></div>' +
      '<div class="search-foot">输入关键词 · Enter 打开第一条</div>' +
    '</div>';
  document.body.appendChild(overlay);
  var input = overlay.querySelector('.search-input');
  var resultsEl = overlay.querySelector('.search-results');
  var pagefindReady = null;
  var debounce = 0;
  var version = 0;
  var previousFocus = null;

  function loadSearch() {
    if (!pagefindReady) {
      // Resolve beside docs.js so both root and GitHub project paths work.
      pagefindReady = import(new URL('pagefind/pagefind.js', scriptURL).href)
        .then(async function (mod) { await mod.init(); return mod; })
        .catch(function (error) { pagefindReady = null; throw error; });
    }
    return pagefindReady;
  }

  function message(text) {
    var p = document.createElement('p');
    p.className = 'search-empty';
    p.textContent = text;
    resultsEl.replaceChildren(p);
  }

  async function search(query, requestVersion) {
    try {
      var pagefind = await loadSearch();
      if (requestVersion !== version) { return; }
      var res = await pagefind.search(query);
      var datas = await Promise.all(res.results.slice(0, 8).map(function (r) { return r.data(); }));
      if (requestVersion !== version) { return; }
      if (!datas.length) { message('没有匹配的页面。'); return; }
      resultsEl.replaceChildren();
      datas.forEach(function (d) {
        var a = document.createElement('a');
        a.href = d.url;
        var title = document.createElement('b');
        title.textContent = d.meta && d.meta.title ? d.meta.title : d.url;
        var excerpt = document.createElement('span');
        // Pagefind escapes excerpts and adds its own <mark> highlights.
        excerpt.innerHTML = d.excerpt || '';
        a.append(title, excerpt);
        resultsEl.appendChild(a);
      });
    } catch (_) {
      if (requestVersion === version) { message('暂时无法加载搜索，请稍后重试。也可以从文档目录查找。'); }
    }
  }

  function open() {
    previousFocus = document.activeElement;
    version += 1;
    overlay.classList.add('is-open');
    document.body.classList.add('search-open');
    input.value = '';
    resultsEl.textContent = '';
    input.focus();
    var openedVersion = version;
    loadSearch().catch(function () {
      if (openedVersion === version) { message('暂时无法加载搜索，请稍后重试。也可以从文档目录查找。'); }
    });
  }
  function close() {
    version += 1;
    window.clearTimeout(debounce);
    overlay.classList.remove('is-open');
    document.body.classList.remove('search-open');
    if (previousFocus) { previousFocus.focus(); }
  }

  searchBtn.addEventListener('click', open);
  overlay.querySelector('.search-close').addEventListener('click', close);
  overlay.addEventListener('click', function (e) { if (e.target === overlay) { close(); } });
  document.addEventListener('keydown', function (e) {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); overlay.classList.contains('is-open') ? close() : open(); }
    if (!overlay.classList.contains('is-open')) { return; }
    if (e.key === 'Escape') { close(); }
    if (e.key === 'Tab') {
      var focusable = overlay.querySelectorAll('button, input, a[href]');
      var firstItem = focusable[0];
      var lastItem = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === firstItem) { e.preventDefault(); lastItem.focus(); }
      if (!e.shiftKey && document.activeElement === lastItem) { e.preventDefault(); firstItem.focus(); }
    }
    if (e.key === 'Enter' && !e.isComposing && e.target === input) {
      var first = resultsEl.querySelector('a');
      if (first) { location.href = first.getAttribute('href'); }
    }
  });

  input.addEventListener('input', function () {
    version += 1;
    var requestVersion = version;
    window.clearTimeout(debounce);
    var query = input.value.trim();
    if (!query) { resultsEl.textContent = ''; return; }
    message('正在搜索…');
    debounce = window.setTimeout(function () {
      search(query, requestVersion);
    }, 160);
  });
})();
