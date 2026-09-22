"""Check local href/src targets in a generated Hugo site."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "site" / "public"
BASE_PATH = "/quickly_access_to_deeplearning/"


class ReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get('id'):
            self.ids.add(attributes['id'])
        for name in ("href", "src"):
            value = attributes.get(name)
            if value:
                self.references.append(value)


def resolve_reference(page: Path, reference: str) -> Path | None:
    if reference.startswith(("http://", "https://", "mailto:", "data:")):
        return None

    target_text = unquote(urlparse(reference).path)
    if not target_text:
        return page
    if target_text.startswith(BASE_PATH):
        target = PUBLIC / target_text.removeprefix(BASE_PATH)
    elif target_text.startswith("/"):
        target = PUBLIC / target_text.removeprefix("/")
    else:
        target = (page.parent / target_text).resolve()
        target.relative_to(PUBLIC.resolve())

    if target.is_dir():
        target /= "index.html"
    return target


def main() -> None:
    if not PUBLIC.is_dir():
        raise SystemExit("site/public does not exist; run the Hugo build first")

    failures: list[str] = []
    pages = sorted(PUBLIC.rglob("*.html"))
    parsed = {}
    for page in pages:
        parser = ReferenceParser()
        parser.feed(page.read_text(encoding="utf-8"))
        parsed[page.resolve()] = parser
    for page in pages:
        parser = parsed[page.resolve()]
        for reference in parser.references:
            try:
                target = resolve_reference(page, reference)
            except ValueError:
                failures.append(f"{page.relative_to(PUBLIC)} -> {reference} (outside site)")
                continue
            if target is not None and not target.exists():
                failures.append(f"{page.relative_to(PUBLIC)} -> {reference}")
            fragment = unquote(urlparse(reference).fragment)
            destination = parsed.get(target.resolve()) if target is not None else None
            if fragment and destination is not None and fragment not in destination.ids:
                failures.append(f"{page.relative_to(PUBLIC)} -> {reference} (missing heading)")

    if failures:
        raise SystemExit("broken site references:\n" + "\n".join(failures))
    print(f"checked {len(pages)} HTML pages; all local files and heading anchors exist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=PUBLIC, help='generated site directory')
    PUBLIC = parser.parse_args().site.resolve()
    main()
