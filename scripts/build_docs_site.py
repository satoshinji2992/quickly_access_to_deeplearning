"""Prepare repository Markdown as Hugo content without duplicating source docs."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
CONTENT = SITE / "content" / "docs"
DOC_ASSETS = SITE / "static" / "assets" / "docs"
REPOSITORY = "https://github.com/satoshinji2992/quickly_access_to_deeplearning"
LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)\n]+)\)")


def yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def read_entries() -> list[dict[str, str]]:
    data = json.loads((SITE / "data" / "docs.json").read_text(encoding="utf-8"))
    entries: list[dict[str, str]] = []
    for group in data["groups"]:
        for item in group["items"]:
            entry = dict(item)
            entry["group"] = group["title"]
            entries.append(entry)
    return entries


def github_url(path: Path, anchor: str = "") -> str:
    relative = path.resolve().relative_to(ROOT).as_posix()
    return f"{REPOSITORY}/blob/main/{quote(relative, safe='/')}{anchor}"


def prepare() -> None:
    entries = read_entries()
    source_to_entry = {(ROOT / entry["source"]).resolve(): entry for entry in entries}

    for directory in (CONTENT.parent, DOC_ASSETS):
        if directory.exists():
            shutil.rmtree(directory)
    CONTENT.mkdir(parents=True)
    DOC_ASSETS.mkdir(parents=True)

    copied_assets: dict[Path, str] = {}

    for index, entry in enumerate(entries):
        source = (ROOT / entry["source"]).resolve()
        if not source.is_file() or ROOT not in source.parents:
            raise FileNotFoundError(entry["source"])
        body = source.read_text(encoding="utf-8")
        body = re.sub(r"\A#\s+[^\n]+\n+", "", body, count=1)

        def replace_link(match: re.Match[str]) -> str:
            image, label, raw_target = match.groups()
            target = raw_target.strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                return match.group(0)
            if " " in target and not target.startswith("<"):
                target = target.split(" ", 1)[0]
            target = target.strip("<>")
            path_text, separator, fragment = target.partition("#")
            anchor = f"#{fragment}" if separator else ""
            resolved = (source.parent / path_text).resolve()

            if image and resolved.is_file():
                if resolved not in copied_assets:
                    relative = resolved.relative_to(ROOT).as_posix()
                    digest = hashlib.sha1(relative.encode()).hexdigest()[:10]
                    filename = f"{digest}-{resolved.name}"
                    shutil.copy2(resolved, DOC_ASSETS / filename)
                    copied_assets[resolved] = filename
                return f"![{label}](../../assets/docs/{quote(copied_assets[resolved])})"

            if resolved in source_to_entry:
                linked = source_to_entry[resolved]
                return f"[{label}](../{linked['slug']}/{anchor})"
            if resolved.is_dir() and (resolved / "README.md").resolve() in source_to_entry:
                linked = source_to_entry[(resolved / "README.md").resolve()]
                return f"[{label}](../{linked['slug']}/{anchor})"
            if ROOT in resolved.parents and resolved.exists():
                return f"[{label}]({github_url(resolved, anchor)})"
            return match.group(0)

        body = LINK.sub(replace_link, body)
        previous = entries[index - 1] if index else None
        following = entries[index + 1] if index + 1 < len(entries) else None
        metadata = [
            "---",
            f"title: {yaml_string(entry['title'])}",
            f"group: {yaml_string(entry['group'])}",
            f"slug: {yaml_string(entry['slug'])}",
            f"weight: {index + 1}",
            f"source_url: {yaml_string(github_url(source))}",
        ]
        if previous:
            metadata.extend((
                f"prev_title: {yaml_string(previous['title'])}",
                f"prev_url: {yaml_string('docs/' + previous['slug'] + '/')}",
            ))
        if following:
            metadata.extend((
                f"next_title: {yaml_string(following['title'])}",
                f"next_url: {yaml_string('docs/' + following['slug'] + '/')}",
            ))
        metadata.extend(("---", ""))
        (CONTENT / f"{entry['slug']}.md").write_text(
            "\n".join(metadata) + body,
            encoding="utf-8",
        )

    print(f"prepared {len(entries)} documents and {len(copied_assets)} images")


if __name__ == "__main__":
    prepare()
