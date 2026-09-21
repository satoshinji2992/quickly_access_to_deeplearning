/* 符号悬停：为 KaTeX 公式里的符号、行内 shape 记号和配图标注提供中文解释。
 * 词典在 symbol-data.js（window.DLSym）；本文件只负责匹配与提示框。 */
(function () {
  'use strict';
  const data = window.DLSym || {};
  const body = document.body;
  const content = document.querySelector('.markdown-body');
  if (!content || !data.scopes) { return; }

  const group = body.getAttribute('data-symscope') || '';
  const scopeKey = group.indexOf('Block 1') === 0 ? 'b1'
    : group.indexOf('Block 2') === 0 ? 'b2'
      : group.indexOf('Block 3') === 0 ? 'b3' : null;
  const dict = Object.assign({}, data.common || {}, scopeKey ? data.scopes[scopeKey] : {});
  const overrides = (data.pageOverrides || {})[body.getAttribute('data-slug')] || {};
  const subGloss = data.subGloss || {};
  const shapeGloss = data.shapeGloss || {};

  const escapeHtml = (text) => text.replace(/[&<>"]/g, (ch) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]
  ));
  const card = (title, text, extra) => {
    let html = '<strong>' + escapeHtml(title) + '</strong><span>' + escapeHtml(text) + '</span>';
    if (extra) { html += '<em>' + escapeHtml(extra) + '</em>'; }
    return html;
  };

  const entries = [];
  const register = (el, html, label) => {
    if (el.dataset.symId) { return; }
    el.classList.add('sym-hot');
    el.setAttribute('aria-label', label);
    el.dataset.symId = String(entries.length);
    entries.push({ el: el, html: html });
  };

  /* ---------- 提示框 ---------- */
  const tip = document.createElement('div');
  tip.className = 'sym-tip';
  tip.setAttribute('role', 'tooltip');
  tip.hidden = true;
  body.appendChild(tip);
  let pinned = false;

  const show = (target) => {
    const item = entries[Number(target.dataset.symId)];
    if (!item) { return; }
    tip.innerHTML = item.html;
    tip.hidden = false;
    tip.classList.add('is-open');
    const rect = target.getBoundingClientRect();
    const width = tip.offsetWidth;
    const height = tip.offsetHeight;
    let left = rect.left + rect.width / 2 - width / 2;
    left = Math.max(10, Math.min(left, window.innerWidth - width - 10));
    let top = rect.top - height - 10;
    if (top < 70) { top = rect.bottom + 10; }
    top = Math.max(70, Math.min(top, window.innerHeight - height - 12));
    tip.style.left = Math.round(left) + 'px';
    tip.style.top = Math.round(top) + 'px';
  };
  const hide = () => {
    if (pinned) { return; }
    tip.classList.remove('is-open');
    tip.hidden = true;
  };

  document.addEventListener('mouseover', (event) => {
    const target = event.target.closest('.sym-hot');
    if (target) { show(target); } else { hide(); }
  });
  document.addEventListener('focusin', (event) => {
    const target = event.target.closest('.sym-hot');
    if (target) { show(target); }
  });
  document.addEventListener('focusout', hide);
  document.addEventListener('click', (event) => {
    const target = event.target.closest('.sym-hot');
    if (target) {
      pinned = target.dataset.symId === tip.dataset.owner ? !pinned : true;
      tip.dataset.owner = target.dataset.symId;
      show(target);
      if (!pinned) { hide(); }
    } else {
      pinned = false;
      hide();
    }
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') { pinned = false; hide(); }
  });
  window.addEventListener('scroll', hide, { passive: true });
  window.addEventListener('resize', hide);

  /* ---------- KaTeX 公式符号 ---------- */
  const LEAF = /^m(ord|op|rel|bin|open|close)$/;
  const SKIP = /(strut|pstrut|vlist|mspace|accent-body|mtight)/;

  const ownText = (el) => {
    let text = '';
    for (const node of el.childNodes) {
      if (node.nodeType === 3) { text += node.textContent; }
    }
    return text.replace(/​/g, '').trim();
  };
  const siblingSub = (leaf) => {
    let el = leaf.nextElementSibling || (leaf.parentElement && leaf.parentElement.nextElementSibling);
    for (; el; el = el.nextElementSibling) {
      if (el.classList.contains('msupsub')) { return el; }
    }
    return null;
  };
  const firstTight = (box) => {
    const tight = box.querySelector('.mtight');
    return tight ? tight.textContent.replace(/​/g, '').trim() : '';
  };

  const SUP_T = ['ᵀ（转置）', '矩阵转置：行列互换。QKᵀ 的第 i 行第 j 列 = Q 的第 i 行与 K 的第 j 行的点积。'];
  const lookup = (leaf, text) => {
    const subBox = siblingSub(leaf);
    const sub = subBox ? firstTight(subBox) : '';
    if (leaf.closest('.mord.accent')) {
      const entry = overrides['hat_' + text] || dict['hat_' + text] || dict['ŷ'];
      if (entry) { return [entry, '']; }
    }
    if (overrides[text + '_' + sub] || dict[text + '_' + sub]) {
      return [overrides[text + '_' + sub] || dict[text + '_' + sub], ''];
    }
    if (overrides[text] || dict[text]) {
      return [overrides[text] || dict[text], subGloss[sub] || ''];
    }
    if (sub && subGloss[sub]) {
      return [[text, subGloss[sub]], ''];
    }
    return null;
  };

  const decorateKatex = (katex) => {
    const htmlPart = katex.querySelector('.katex-html');
    if (!htmlPart || katex.dataset.symDone) { return; }
    katex.dataset.symDone = '1';
    htmlPart.querySelectorAll('span').forEach((leaf) => {
      if (leaf.dataset.symId) { return; }
      if (!Array.from(leaf.classList).some((name) => LEAF.test(name))) { return; }
      if (SKIP.test(leaf.className)) { return; }
      const inSub = leaf.closest('.msupsub');
      const text = ownText(leaf);
      if (!text || text.length > 24) { return; }
      if (inSub) {
        const label = text === 'T' ? SUP_T : (subGloss[text] ? [text, subGloss[text]] : null);
        if (label) {
          register(leaf, card(label[0], label[1]), label[0] + '：' + label[1]);
        }
        return;
      }
      const found = lookup(leaf, text);
      if (found) {
        register(leaf, card(found[0][0], found[0][1], found[1]), found[0][0] + '：' + found[0][1]);
      }
    });
  };

  /* ---------- 行内 shape 记号 ---------- */
  const SHAPE_SPLIT = /(?<![.\w])([A-Za-z][A-Za-z0-9_]*)(?![.\w])/g;
  const decorateCode = (code) => {
    if (code.dataset.symDone || code.closest('pre')) { return; }
    const text = code.textContent;
    if (text.length > 80 || /(^|[^\w])d[A-Z_]/.test(text)) { return; }
    const tokens = text.match(SHAPE_SPLIT) || [];
    const known = tokens.filter((token) => shapeGloss[token]);
    const tupleLike = /^\s*[([]/.test(text) && /[)\]]\s*$/.test(text);
    if (known.length < 1 || tokens.length > 6) { return; }
    if (known.length < 2 && !(known.length === 1 && tupleLike)) { return; }
    code.dataset.symDone = '1';
    let html = '';
    let last = 0;
    text.replace(SHAPE_SPLIT, (match, token, offset) => {
      html += escapeHtml(text.slice(last, offset));
      if (shapeGloss[token]) {
        html += '<span class="sym-shape" data-sym-text="' + escapeHtml(token) + '">' + escapeHtml(token) + '</span>';
      } else {
        html += escapeHtml(token);
      }
      last = offset + match.length;
      return match;
    });
    html += escapeHtml(text.slice(last));
    code.innerHTML = html;
    code.querySelectorAll('.sym-shape').forEach((el) => {
      const token = el.dataset.symText;
      register(el, card(token, shapeGloss[token]), shapeGloss[token]);
    });
  };

  /* ---------- 调度：KaTeX 异步渲染，用 MutationObserver 兜住 ---------- */
  let scheduled = false;
  const pass = () => {
    scheduled = false;
    content.querySelectorAll('.katex').forEach(decorateKatex);
    content.querySelectorAll('code').forEach(decorateCode);
  };
  const schedule = () => {
    if (scheduled) { return; }
    scheduled = true;
    window.requestAnimationFrame(pass);
  };
  new MutationObserver(schedule).observe(content, { childList: true, subtree: true });
  schedule();

  /* ---------- QA 深链：?symtip=<标注文字> 固定显示一个提示 ---------- */
  const wanted = new URLSearchParams(window.location.search).get('symtip');
  if (wanted) {
    window.setTimeout(() => {
      const target = entries.map((item) => item.el).find((el) => (
        el.textContent.trim() === wanted || (el.getAttribute('aria-label') || '').indexOf(wanted) === 0
      ));
      if (target) {
        pinned = true;
        tip.dataset.owner = target.dataset.symId;
        show(target);
      }
    }, 700);
  }
})();
