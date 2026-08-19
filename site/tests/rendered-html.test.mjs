import assert from "node:assert/strict";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("https://deep-learning.example/", { headers: { accept: "text/html", host: "deep-learning.example", "x-forwarded-proto": "https" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the finished project page", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<html lang="zh-CN">/);
  assert.match(html, /<title>快速入门深度学习｜从 NumPy 到 MiniMind<\/title>/);
  assert.match(html, /把深度学习/);
  assert.match(html, /真正跑通/);
  assert.match(html, /先把神经网络拆开看/);
  assert.match(html, /再让像素流过 ResNet/);
  assert.match(html, /最后拼出一个小型 Transformer/);
  assert.match(html, /Ran 43 checks/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|SkeletonPreview/);
});

test("emits share metadata from the request host", async () => {
  const html = await (await render()).text();
  assert.match(html, /property="og:title" content="快速入门深度学习｜从 NumPy 到 MiniMind"/);
  assert.match(html, /property="og:image" content="https:\/\/deep-learning\.example\/og\.png"/);
  assert.match(html, /name="twitter:card" content="summary_large_image"/);
  assert.match(html, /name="twitter:image" content="https:\/\/deep-learning\.example\/og\.png"/);
});
