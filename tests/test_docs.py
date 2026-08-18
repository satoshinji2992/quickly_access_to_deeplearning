"""Repository-wide checks for local Markdown links and image references."""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def markdown_files():
    for path in ROOT.rglob("*.md"):
        if ".git" not in path.parts:
            yield path


def local_targets(path: Path):
    text = path.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        for match in MARKDOWN_LINK.finditer(line):
            raw_target = match.group(1).strip()
            if raw_target.startswith("<") and ">" in raw_target:
                raw_target = raw_target[1 : raw_target.index(">")]
            else:
                # Markdown permits an optional title after a whitespace separator.
                raw_target = raw_target.split(maxsplit=1)[0]
            target = unquote(raw_target.split("#", 1)[0])
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            yield line_number, target


class DocumentationTests(unittest.TestCase):
    def test_all_local_markdown_targets_exist(self):
        broken = []
        for document in markdown_files():
            for line_number, target in local_targets(document):
                destination = (document.parent / target).resolve()
                if not destination.exists():
                    broken.append(
                        f"{document.relative_to(ROOT)}:{line_number} -> {target}"
                    )
        self.assertEqual([], broken, "broken local links:\n" + "\n".join(broken))

    def test_active_docs_do_not_use_placeholder_image_alt_text(self):
        offenders = []
        for directory in (ROOT / "chapters", ROOT / "exercises"):
            for document in directory.rglob("*.md"):
                for line_number, line in enumerate(
                    document.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if "![alt text]" in line:
                        offenders.append(
                            f"{document.relative_to(ROOT)}:{line_number}"
                        )
        self.assertEqual([], offenders)

    def test_markdown_fences_and_display_math_are_balanced(self):
        failures = []
        for document in markdown_files():
            text = document.read_text(encoding="utf-8")
            if text.count("$$") % 2:
                failures.append(f"{document.relative_to(ROOT)}: unbalanced $$")
            fence_count = sum(
                line.startswith("```") for line in text.splitlines()
            )
            if fence_count % 2:
                failures.append(
                    f"{document.relative_to(ROOT)}: unbalanced fenced code block"
                )
        self.assertEqual([], failures, "\n".join(failures))

    def test_display_math_does_not_contain_cjk_sentence_punctuation(self):
        """Keep prose punctuation outside KaTeX/LaTeX display blocks."""
        offenders = []
        for document in markdown_files():
            chunks = document.read_text(encoding="utf-8").split("$$")
            for index in range(1, len(chunks), 2):
                if any(mark in chunks[index] for mark in "，。；："):
                    line_number = 1 + sum(
                        chunk.count("\n") for chunk in chunks[:index]
                    )
                    offenders.append(
                        f"{document.relative_to(ROOT)}:{line_number}"
                    )
        self.assertEqual(
            [], offenders, "CJK punctuation inside display math:\n" + "\n".join(offenders)
        )


if __name__ == "__main__":
    unittest.main()
