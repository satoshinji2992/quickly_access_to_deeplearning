"""Checks for the Hugo project site deployed on GitHub Pages."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
HOMEPAGE = SITE / "layouts" / "index.html"


class ProjectSiteTests(unittest.TestCase):
    def test_homepage_local_assets_exist(self):
        html = HOMEPAGE.read_text(encoding="utf-8")
        references = re.findall(r'(?:src|href)="([^"]+)"', html)
        missing = []
        for reference in references:
            if reference.startswith(("http://", "https://", "#")):
                continue
            target_text = reference.split("#", 1)[0]
            if not target_text or target_text.startswith("./docs/"):
                continue
            target = (SITE / "static" / target_text.removeprefix("./")).resolve()
            if not target.exists():
                missing.append(reference)
        self.assertEqual([], missing)

    def test_copy_avoids_promotional_boilerplate(self):
        html = HOMEPAGE.read_text(encoding="utf-8")
        phrases = (
            "真正跑通",
            "拆开黑盒",
            "一眼看懂",
            "API 巡礼",
            "不止能运行",
            "轻松掌握",
            "开启探索",
            "赋能",
        )
        self.assertEqual([], [phrase for phrase in phrases if phrase in html])

    def test_docs_catalog_sources_and_slugs(self):
        catalog = json.loads((SITE / "data" / "docs.json").read_text(encoding="utf-8"))
        items = [item for group in catalog["groups"] for item in group["items"]]
        self.assertEqual(25, len(items))
        self.assertEqual(len(items), len({item["slug"] for item in items}))
        self.assertEqual([], [item["source"] for item in items if not (ROOT / item["source"]).is_file()])
        self.assertTrue((SITE / "layouts" / "_default" / "single.html").is_file())
        self.assertTrue((SITE / "static" / "docs.css").is_file())

    def test_pages_workflow_and_public_metadata(self):
        html = HOMEPAGE.read_text(encoding="utf-8")
        workflow = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
        self.assertIn("hugo --source site --minify", workflow)
        self.assertIn("python scripts/check_site_links.py", workflow)
        self.assertIn("path: site/public", workflow)
        self.assertIn("actions/deploy-pages@v4", workflow)
        self.assertIn(
            "https://satoshinji2992.github.io/quickly_access_to_deeplearning/",
            (ROOT / "README.md").read_text(encoding="utf-8"),
        )
        self.assertIn("https://satoshinji2992.github.io/quickly_access_to_deeplearning/assets/og.png", html)


if __name__ == "__main__":
    unittest.main()
