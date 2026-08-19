"""Repository-wide checks for local Markdown links and image references."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def markdown_files():
    for path in ROOT.rglob("*.md"):
        if path.is_relative_to(ROOT / "site" / "content"):
            continue
        if ".git" not in path.parts and "node_modules" not in path.parts:
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
    def test_narrative_chapters_link_every_implementation_section(self):
        catalog = json.loads(
            (ROOT / "site" / "data" / "docs.json").read_text(encoding="utf-8")
        )
        missing = []
        for group in catalog["groups"]:
            if not group["title"].startswith("Block"):
                continue
            chapter = (ROOT / group["items"][0]["source"]).resolve()
            linked = {
                (chapter.parent / target).resolve()
                for _, target in local_targets(chapter)
            }
            for item in group["items"][1:]:
                source = (ROOT / item["source"]).resolve()
                if source not in linked:
                    missing.append(f"{chapter.relative_to(ROOT)} -> {item['source']}")
        self.assertEqual([], missing, "missing narrative links:\n" + "\n".join(missing))

    def test_narrative_chapters_avoid_assignment_and_promotional_tone(self):
        phrases = ("请你", "提交作业", "轻松掌握", "一眼看懂", "从零到精通", "赋能")
        offenders = []
        for chapter in sorted((ROOT / "chapters").glob("*.md")):
            text = chapter.read_text(encoding="utf-8")
            for phrase in phrases:
                if phrase in text:
                    offenders.append(f"{chapter.relative_to(ROOT)}: {phrase}")
        self.assertEqual([], offenders)

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

    def test_widget_mounts_reference_registered_widgets(self):
        """data-widget markers must match the panel.js registry and files."""
        registry_file = ROOT / "site" / "static" / "widgets" / "panel.js"
        registry = registry_file.read_text(encoding="utf-8")
        registered = set(re.findall(r"'([a-z0-9-]+)':\s*\{ file:", registry))
        self.assertTrue(registered, "panel.js registry not parseable")
        offenders = []
        for document in markdown_files():
            for line_number, line in enumerate(
                document.read_text(encoding="utf-8").splitlines(), start=1
            ):
                for widget_id in re.findall(r'data-widget="([a-z0-9-]+)"', line):
                    if widget_id not in registered:
                        offenders.append(
                            f"{document.relative_to(ROOT)}:{line_number}: {widget_id}"
                        )
        for widget_id in sorted(registered):
            entry = re.search(
                r"'" + widget_id + r"':\s*\{ file: '([^']+)'", registry
            )
            if entry and not (ROOT / "site" / "static" / "widgets" / entry.group(1)).exists():
                offenders.append(f"panel.js registry: missing file {entry.group(1)}")
        self.assertEqual(
            [], offenders, "unknown or broken widget references:\n" + "\n".join(offenders)
        )

    def test_math_survives_markdown_and_katex(self):
        """Patterns known to break goldmark passthrough or KaTeX."""
        offenders = []
        for document in markdown_files():
            text = document.read_text(encoding="utf-8")
            chunks = text.split("$$")
            for index in range(1, len(chunks), 2):
                body = chunks[index]
                line_number = 1 + sum(
                    chunk.count("\n") for chunk in chunks[:index]
                )
                for offset, line in enumerate(body.splitlines()):
                    # A lone = or - line inside display math is parsed as a
                    # setext heading underline before passthrough sees the block.
                    if line.strip() in ("=", "-"):
                        offenders.append(
                            f"{document.relative_to(ROOT)}:{line_number + offset}: lone {line.strip()!r} line inside $$"
                        )
            for match in re.finditer(r"\\text\{([^{}]*)\}", text):
                # KaTeX rejects unescaped underscores inside \text{}.
                if re.search(r"(?<!\\)_", match.group(1)):
                    offenders.append(
                        f"{document.relative_to(ROOT)}: unescaped _ inside \\text{{{match.group(1)}}}"
                    )
            # A control word glued to @ (e.g. \top@) lexes as one undefined command.
            for match in re.finditer(r"\\[a-zA-Z]+@", text):
                offenders.append(
                    f"{document.relative_to(ROOT)}: control word followed by @ in {match.group(0)}"
                )
        self.assertEqual(
            [], offenders, "math patterns that break rendering:\n" + "\n".join(offenders)
        )


if __name__ == "__main__":
    unittest.main()
