/* Interactive widget panel: registry, lazy loading, right-side dock.
 *
 * Contract for widget files (w-<id>.js):
 *   DLWidget.register(id, mountFn)
 *     - mountFn(container) is called each time the widget's tab is opened.
 *     - container is a fresh <div class="widget-instance">; mountFn builds
 *       its DOM inside and returns an optional dispose() called on close.
 *   Panels are styled by widgets.css; keep visuals in the site's paper style.
 */
(function () {
  'use strict';

  var REGISTRY = {
    'linear-fit': { file: 'w-linear-fit.js', title: '线性回归：调 a 与 b' },
    'conv-sweep': { file: 'w-conv-sweep.js', title: '卷积：拖动核扫过图像' },
    'kv-cache': { file: 'w-kv-cache.js', title: 'KV Cache：逐步解码' },
  };

  var base = (function () {
    var src = (document.currentScript && document.currentScript.src) || '';
    var cut = src.lastIndexOf('/');
    return cut >= 0 ? src.slice(0, cut + 1) : '';
  })();

  var loaded = {};   // id -> true once the script file has been injected
  var mounted = {};  // id -> { dispose }
  var panel = null;
  var tabsBar = null;
  var body = null;
  var openOrder = [];

  function ensurePanel() {
    if (panel) { return; }
    panel = document.createElement('aside');
    panel.className = 'widget-panel';
    panel.setAttribute('aria-label', '交互演示面板');
    var bar = document.createElement('div');
    bar.className = 'widget-panel-bar';
    tabsBar = document.createElement('div');
    tabsBar.className = 'widget-tabs';
    var close = document.createElement('button');
    close.type = 'button';
    close.className = 'widget-panel-close';
    close.setAttribute('aria-label', '关闭面板');
    close.textContent = '✕';
    close.addEventListener('click', function () { hidePanel(); });
    bar.appendChild(tabsBar);
    bar.appendChild(close);
    body = document.createElement('div');
    body.className = 'widget-body';
    panel.appendChild(bar);
    panel.appendChild(body);
    document.body.appendChild(panel);
  }

  function hidePanel() {
    if (panel) { panel.hidden = true; }
    document.querySelectorAll('.widget-trigger.is-active').forEach(function (b) {
      b.classList.remove('is-active');
    });
  }

  function tabFor(id) {
    var tab = document.createElement('button');
    tab.type = 'button';
    tab.className = 'widget-tab';
    tab.textContent = REGISTRY[id].title;
    tab.addEventListener('click', function () { activate(id); });
    tab.dataset.widget = id;
    return tab;
  }

  function activate(id) {
    if (!REGISTRY[id]) { return; }
    ensurePanel();
    panel.hidden = false;
    if (openOrder.indexOf(id) < 0) { openOrder.push(id); }

    // Unmount the currently visible widget.
    Object.keys(mounted).forEach(function (key) {
      if (key !== id && mounted[key]) {
        if (mounted[key].dispose) { mounted[key].dispose(); }
        delete mounted[key];
        var host = body.querySelector('[data-host="' + key + '"]');
        if (host) { host.remove(); }
      }
    });

    if (!tabsBar.querySelector('[data-widget="' + id + '"]')) {
      tabsBar.appendChild(tabFor(id));
    }
    tabsBar.querySelectorAll('.widget-tab').forEach(function (t) {
      t.classList.toggle('is-active', t.dataset.widget === id);
    });

    var host = body.querySelector('[data-host="' + id + '"]');
    if (!host) {
      host = document.createElement('div');
      host.className = 'widget-instance';
      host.dataset.host = id;
      body.appendChild(host);
      host.appendChild(document.createComment('loading ' + id));
    }

    if (loaded[id] && window.DLWidget._impl[id]) {
      mountInto(id, host);
    } else if (!loaded[id]) {
      loaded[id] = true;
      var script = document.createElement('script');
      script.src = base + REGISTRY[id].file;
      script.onerror = function () {
        host.textContent = '组件加载失败：' + REGISTRY[id].file;
      };
      document.head.appendChild(script);
      // The widget file calls DLWidget.register() at load end; poll briefly.
      waitFor(id, Date.now(), host);
    }
  }

  function waitFor(id, start, host) {
    if (window.DLWidget._impl[id]) {
      mountInto(id, host);
    } else if (Date.now() - start < 8000) {
      window.setTimeout(function () { waitFor(id, start, host); }, 60);
    }
  }

  function mountInto(id, host) {
    if (mounted[id]) { return; }
    while (host.firstChild) { host.removeChild(host.firstChild); }
    var dispose = window.DLWidget._impl[id](host) || null;
    mounted[id] = { dispose: dispose };
  }

  function makeTrigger(id, label) {
    var trigger = document.createElement('button');
    trigger.type = 'button';
    trigger.className = 'widget-trigger';
    trigger.textContent = '▶ 交互演示' + (label ? ' · ' + label : '');
    trigger.addEventListener('click', function () {
      document.querySelectorAll('.widget-trigger.is-active').forEach(function (b) {
        b.classList.remove('is-active');
      });
      var again = trigger.classList.contains('is-active');
      trigger.classList.add('is-active');
      if (again && panel && !panel.hidden) { hidePanel(); return; }
      activate(id);
      var host = document.querySelector('[data-host="' + id + '"]');
      if (host) { host.scrollIntoView({ block: 'nearest', inline: 'nearest' }); }
    });
    return trigger;
  }

  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.widget-mount').forEach(function (mount) {
      var id = mount.dataset.widget;
      if (!REGISTRY[id]) { return; }
      mount.appendChild(makeTrigger(id, mount.dataset.title || REGISTRY[id].title));
    });
    // Deep link / QA hook: docs/<slug>/?widget=<id> opens that widget.
    var wanted = new URLSearchParams(window.location.search).get('widget');
    if (wanted && REGISTRY[wanted]) { activate(wanted); }
  });

  window.DLWidget = {
    _impl: {},
    register: function (id, mountFn) {
      this._impl[id] = mountFn;
      var host = document.querySelector('[data-host="' + id + '"]');
      if (host) { mountInto(id, host); }
    },
    open: function (id) { activate(id); },
    close: function () { hidePanel(); },
  };
})();
