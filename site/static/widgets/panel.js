/* Interactive widgets: registry + inline lazy mounting in the article flow.
 *
 * Contract for widget files (w-<id>.js):
 *   DLWidget.register(id, mountFn)
 *     - mountFn(container) is called once when the widget scrolls near the
 *       viewport; it builds its DOM inside and returns an optional dispose().
 *   Docs embed <div class="widget-mount" data-widget="<id>" data-title="...">.
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

  var loading = {};   // id -> script injected
  var pending = {};   // id -> [container, ...] waiting for the script
  var mounted = {};   // container -> { id, dispose }

  function mountInto(id, container) {
    var impl = window.DLWidget._impl[id];
    if (!impl || container.dataset.mounted) { return; }
    container.dataset.mounted = '1';
    var frame = document.createElement('div');
    frame.className = 'widget-frame';
    container.appendChild(frame);
    var dispose = impl(frame) || null;
    mounted[container] = { id: id, dispose: dispose };
  }

  function loadWidget(id, container) {
    if (!REGISTRY[id]) { return; }
    if (window.DLWidget._impl[id]) { mountInto(id, container); return; }
    (pending[id] = pending[id] || []).push(container);
    if (loading[id]) { return; }
    loading[id] = true;
    var script = document.createElement('script');
    script.src = base + REGISTRY[id].file;
    script.onerror = function () {
      var note = document.createElement('p');
      note.className = 'wg-note';
      note.textContent = '组件加载失败：' + REGISTRY[id].file;
      container.appendChild(note);
    };
    document.head.appendChild(script);
  }

  function prepare(mount) {
    var id = mount.dataset.widget;
    if (!REGISTRY[id]) { return; }
    if (!mount.id) { mount.id = 'widget-' + id; }
    mount.classList.add('is-ready');
    var head = document.createElement('div');
    head.className = 'widget-head';
    head.appendChild(document.createTextNode('交互演示' + (mount.dataset.title ? ' · ' + mount.dataset.title : '')));
    mount.appendChild(head);
    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            io.disconnect();
            loadWidget(id, mount);
          }
        });
      }, { rootMargin: '600px 0px' });
      io.observe(mount);
    } else {
      loadWidget(id, mount);
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.widget-mount').forEach(prepare);
    // Deep link / QA hook: docs/<slug>/?widget=<id> jumps to that widget and
    // mounts it immediately. Uses a native hash jump (temporarily disabling
    // smooth scrolling) so it lands deterministically.
    var wanted = new URLSearchParams(window.location.search).get('widget');
    if (wanted && REGISTRY[wanted]) {
      var target = document.querySelector('.widget-mount[data-widget="' + wanted + '"]');
      if (target) {
        var root = document.documentElement;
        var smooth = root.style.scrollBehavior;
        root.style.scrollBehavior = 'auto';
        window.location.hash = 'widget-' + wanted;
        root.style.scrollBehavior = smooth;
        loadWidget(wanted, target);
      }
    }
  });

  window.DLWidget = {
    _impl: {},
    register: function (id, mountFn) {
      this._impl[id] = mountFn;
      (pending[id] || []).forEach(function (container) { mountInto(id, container); });
      pending[id] = [];
    },
  };
})();
