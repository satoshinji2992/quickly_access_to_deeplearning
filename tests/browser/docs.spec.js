const { test, expect } = require('@playwright/test');

function collectRenderingErrors(page) {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error' && /katex|parseerror/i.test(message.text())) {
      errors.push(message.text());
    }
  });
  return errors;
}

test('the homepage keeps its title and fits a phone screen', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.locator('h1')).toHaveText('深度学习从零实现');
  const dimensions = await page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
  expect(dimensions[0]).toBeLessThanOrEqual(dimensions[1]);
});

test('copying a code block twice preserves the exact code', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/docs/linear-regression/');
  const block = page.locator('.markdown-body pre').first();
  const source = await block.locator('code').textContent();
  for (let i = 0; i < 2; i += 1) {
    await block.getByRole('button', { name: '复制代码', exact: true }).click();
    await expect.poll(() => page.evaluate(() => navigator.clipboard.readText())).toBe(source);
  }
});

test('search works from the catalog and accepts typing before the index loads', async ({ page }) => {
  await page.route('**/pagefind/pagefind.js', async route => {
    await new Promise(resolve => setTimeout(resolve, 600));
    await route.continue();
  });
  await page.goto('/docs/');
  await page.getByRole('button', { name: /搜索/ }).click();
  await page.getByRole('searchbox').fill('SwiGLU');
  const results = page.locator('.search-results a');
  await expect(results.first()).toBeVisible();
  await expect(page.locator('.search-results a[href*="swiglu"]')).toBeVisible();
  await page.getByRole('searchbox').fill('无匹配的独特词 xyz987654');
  await expect(page.locator('.search-results')).toContainText('没有匹配');
  await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: /搜索/ })).toBeFocused();
});

test('an older search cannot overwrite a later query or a cleared field', async ({ page }) => {
  await page.route('**/pagefind/pagefind.js', route => route.fulfill({
    contentType: 'text/javascript',
    body: `export async function init() {}
      export async function search(q) {
        await new Promise(r => setTimeout(r, q === 'old' ? 1000 : 10));
        return {results:[{data:async()=>({url:'/docs/'+q+'/',meta:{title:q},excerpt:q})}]};
      }`,
  }));
  await page.goto('/docs/');
  await page.getByRole('button', { name: /搜索/ }).click();
  const input = page.getByRole('searchbox');
  await input.fill('old');
  await page.waitForTimeout(250);
  await input.fill('new');
  await expect(page.locator('.search-results b')).toHaveText('new');
  await page.waitForTimeout(1100);
  await expect(page.locator('.search-results b')).toHaveText('new');
  await input.fill('old');
  await page.waitForTimeout(250);
  await input.fill('');
  await page.waitForTimeout(1100);
  await expect(page.locator('.search-results')).toBeEmpty();
});

test('phone readers can open the page outline and dismiss search by touch', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/docs/conv2d-im2col/');
  const outline = page.locator('.mobile-toc');
  await outline.getByText('本页目录', { exact: true }).click();
  await expect(outline.locator('a').first()).toBeVisible();
  await page.getByRole('button', { name: /搜索/ }).click();
  await page.getByRole('button', { name: '关闭搜索' }).click();
  await expect(page.getByRole('dialog')).not.toBeVisible();
  const dimensions = await page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
  expect(dimensions[0]).toBeLessThanOrEqual(dimensions[1]);
});

test('every catalog page renders and mounts its interactive illustrations', async ({ page }) => {
  test.setTimeout(180000);
  const catalog = require('../../site/data/docs.json');
  const errors = collectRenderingErrors(page);
  for (const group of catalog.groups) {
    for (const entry of group.items) {
      const response = await page.goto('/docs/' + entry.slug + '/');
      expect(response.status(), entry.slug).toBe(200);
      await expect(page.locator('.docs-heading h1')).toHaveText(entry.page_title || entry.title);
      expect(await page.evaluate(() => typeof window.katex?.render), entry.slug + ': KaTeX loaded').toBe('function');
      expect(await page.evaluate(() => typeof window.renderMathInElement), entry.slug + ': auto-render loaded').toBe('function');
      await expect(page.locator('.katex-error'), entry.slug + ': formula rendering').toHaveCount(0);
      const widgets = page.locator('.widget-mount');
      for (let i = 0; i < await widgets.count(); i += 1) {
        const widget = widgets.nth(i);
        await widget.evaluate(node => {
          let parent = node.parentElement;
          while (parent) {
            if (parent.tagName === 'DETAILS') parent.open = true;
            parent = parent.parentElement;
          }
          node.scrollIntoView({ behavior: 'instant', block: 'center' });
        });
        await expect(widget.locator('.widget-frame'), entry.slug + ': ' + await widget.getAttribute('data-widget')).not.toBeEmpty();
      }
      expect(errors, entry.slug).toEqual([]);
    }
  }
});

test('formula checks catch parse errors even when auto-render leaves the source text', async ({ page }) => {
  await page.goto('/docs/linear-regression/');
  const errors = collectRenderingErrors(page);
  await page.evaluate(() => {
    const formula = document.createElement('div');
    formula.textContent = String.raw`$$\notARealTeXCommand{x}$$`;
    document.body.appendChild(formula);
    window.renderMathInElement(formula);
  });
  await expect.poll(() => errors.length).toBeGreaterThan(0);
  expect(errors.join('\n')).toMatch(/katex|parseerror/i);
});
