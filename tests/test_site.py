"""Checks for the static GitHub Pages project site."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


class ProjectSiteTests(unittest.TestCase):
    def test_local_assets_exist(self):
        html = (SITE / "index.html").read_text(encoding="utf-8")
        references = re.findall(r'(?:src|href)="([^"]+)"', html)
        missing = []
        for reference in references:
            if reference.startswith(("http://", "https://", "#")):
                continue
            target = (SITE / reference.split("#", 1)[0]).resolve()
            if not target.exists():
                missing.append(reference)
        self.assertEqual([], missing)

    def test_copy_avoids_promotional_boilerplate(self):
        html = (SITE / "index.html").read_text(encoding="utf-8")
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

    def test_pages_workflow_and_public_metadata(self):
        html = (SITE / "index.html").read_text(encoding="utf-8")
        workflow = ROOT / ".github" / "workflows" / "pages.yml"
        self.assertTrue(workflow.exists())
        self.assertIn("actions/deploy-pages@v4", workflow.read_text(encoding="utf-8"))
        self.assertIn(
            "https://satoshinji2992.github.io/quickly_access_to_deeplearning/",
            (ROOT / "README.md").read_text(encoding="utf-8"),
        )
        self.assertIn("https://satoshinji2992.github.io/quickly_access_to_deeplearning/assets/og.png", html)


if __name__ == "__main__":
    unittest.main()
