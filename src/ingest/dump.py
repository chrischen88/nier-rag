"""Streaming reader for a MediaWiki XML dump."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

CATEGORY_RE = re.compile(r"\[\[\s*Category\s*:\s*([^\]|]+)", re.I)

NS_MAIN, NS_CATEGORY = 0, 14


@dataclass
class RawPage:
    page_id: int
    title: str
    ns: int
    revision_id: int
    timestamp: str
    text: str
    redirect: str | None

    @property
    def categories(self) -> list[str]:
        """Explicit [[Category:...]] tags. Categories added by templates aren't visible offline."""
        seen: dict[str, None] = {}
        for c in CATEGORY_RE.findall(self.text):
            seen[normalize_title(c)] = None
        return list(seen)


def normalize_title(title: str) -> str:
    """MediaWiki title normalization for this wiki (case=first-letter)."""
    t = re.sub(r"[\s_]+", " ", title).strip()
    return t[:1].upper() + t[1:]


def iter_pages(path: Path) -> Iterator[RawPage]:
    for _, el in ET.iterparse(path):
        if not el.tag.endswith("}page"):
            continue
        rev = el.find("{*}revision")
        redirect = el.find("{*}redirect")
        yield RawPage(
            page_id=int(el.find("{*}id").text),
            title=el.find("{*}title").text,
            ns=int(el.find("{*}ns").text),
            revision_id=int(rev.find("{*}id").text),
            timestamp=rev.find("{*}timestamp").text,
            text=rev.find("{*}text").text or "",
            redirect=normalize_title(redirect.get("title")) if redirect is not None else None,
        )
        el.clear()


@dataclass
class Wiki:
    articles: dict[str, RawPage]  # non-redirect main-namespace pages by title
    redirects: dict[str, str]  # redirect title -> target title
    category_parents: dict[str, list[str]]  # category -> its parent categories

    def resolve(self, title: str) -> str | None:
        """Follow redirects (up to 5 hops); None if the target isn't an article."""
        t = normalize_title(title.split("#", 1)[0]) if title.strip() else ""
        for _ in range(5):
            if t in self.articles:
                return t
            if t not in self.redirects:
                return None
            t = self.redirects[t]
        return None

    def aliases(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for src in self.redirects:
            if (dst := self.resolve(src)) and dst != src:
                out.setdefault(dst, []).append(src)
        return out


def load_wiki(path: Path) -> Wiki:
    articles, redirects, parents = {}, {}, {}
    for p in iter_pages(path):
        if p.ns == NS_MAIN:
            if p.redirect:
                redirects[normalize_title(p.title)] = p.redirect
            else:
                articles[normalize_title(p.title)] = p
        elif p.ns == NS_CATEGORY:
            parents[normalize_title(p.title.split(":", 1)[1])] = p.categories
    return Wiki(articles, redirects, parents)
