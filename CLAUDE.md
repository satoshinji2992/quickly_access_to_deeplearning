# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Chinese-language, implementation-first deep learning tutorial (快速入门深度学习). Three blocks: NumPy fundamentals + a mini DL library, a NumPy ResNet on CIFAR-100, and a decoder-only Transformer (RoPE / GQA / KV Cache). The repo doubles as a Hugo static site published to GitHub Pages. All tutorial content is written in Simplified Chinese — write new content in Chinese to match.

## Commands

```bash
# Environment (conda)
conda create -n dl_tutorial python=3.10
conda activate dl_tutorial
pip install -r requirements.txt   # numpy/pandas/matplotlib/sklearn + torch/torchvision

# Tests (no data download needed)
python -m unittest discover -s tests -p 'test_block1.py' -v   # also test_block2.py, test_block3.py
python -m unittest tests.test_docs -v                          # Markdown link/reference checks
python -m unittest tests.test_docs tests.test_site -v          # full doc checks (same as CI)

# Single test
python -m unittest tests.test_block3.SomeClass.test_method -v

# Docs site (Hugo) — build, preview, verify
python scripts/build_docs_site.py
hugo server --source site          # preview at http://localhost:1313/quickly_access_to_deeplearning/
hugo --source site --minify
python scripts/check_site_links.py

# Reference implementations
python solutions/block_01_basics/linear_regression_solution.py
python solutions/block_02_resnet/train_cifar100_solution.py --subset-size 200 --epochs 20 --batch-size 20 --channels 8 16 32 --lr 0.03
python solutions/block_03_transformer/minimind_solution.py
```

Every test command is expected to end with `OK`.

## Architecture

**Two kinds of pages with different jobs** (see CONTRIBUTING.md):
- `chapters/*.md` — the narrative spine, read start to finish. Explains *why* something appears, builds intuition before math.
- `exercises/block_0N/task_XX_*/README.md` — implementation manuals: full shapes, interfaces, commands, expected output, test boundaries. Explains *how* the code works.
- Don't duplicate the same checklist in both places; link between them instead.
- `solutions/` holds reference implementations.

**Code layout:**
- `common/my_dl_lib.py` — the shared mini NumPy DL library (task_02 onward): Linear, ReLU/SiLU/GELU, BatchNorm1D, LayerNorm, Dropout, CrossEntropyLoss, Sequential, and optimizers (SGD/Momentum/Adagrad/RMSProp/Adam/AdamW). Exercise scripts reach it by inserting the repo root into `sys.path` and importing `common`; follow that pattern in new exercise code.
- `data/` and `checkpoints/` are gitignored runtime artifacts (CIFAR-100, trained weights).
- `tests/` — unittest suite covering data isolation, gradients, shapes, checkpoint round-trip, and causality; it does not measure full training accuracy.

**Docs pipeline:** repo Markdown is the single source of truth. `scripts/build_docs_site.py` reads the catalog `site/data/docs.json` (groups → items with `source`/`slug`; must stay 25 items with unique slugs), converts each source file into `site/content/docs/<slug>.md` with Hugo front matter (prev/next), rewrites internal links, and copies images to `site/static/assets/docs/`. Generated dirs (`site/content`, `site/public`, `site/static/assets/docs`) are never committed. CI (`.github/workflows/pages.yml`) reruns the doc tests + Hugo build on pushes touching chapters/exercises READMEs/site/scripts, then deploys.

**Content gates enforced by tests:**
- `tests/test_docs.py` — every local Markdown link/image target must resolve; each Block chapter must link to all of its exercise READMEs.
- `tests/test_site.py` — homepage assets must exist; the homepage must not contain promotional boilerplate (真正跑通, 拆开黑盒, 一眼看懂, 轻松掌握, 赋能, etc.); docs.json catalog integrity.

## Writing conventions (from CONTRIBUTING.md)

When adding or rewriting a section, follow this progression — as narrative flow, not a rigid template of seven headings:
1. 承接 — where does the previous method fail on a concrete example?
2. 最小例子 — something hand-computable or drawable first
3. 数学 — every symbol already introduced earlier
4. shape — single sample → batch → the repo's axis order
5. 代码落点 — link to the real file (forward / cache / backward / state)
6. 可观察现象 — concrete numbers, curves, properties, or failure modes; not vague "understand/master" goals
7. 过渡 — the gap left behind leads into the next section

Tone: describe "why it's needed" and "what you'll observe at runtime" rather than issuing commands; no homework/assignments; no promotional claims; don't present smoke tests or short training runs as real model capability; analogies build first intuition, then return to formulas, shapes, or code.
