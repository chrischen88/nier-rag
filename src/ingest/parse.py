"""Wikitext cleaning with mwparserfromhell (SPEC §5.3).

Pipeline per page:
  1. Pull the first infobox out as structured metadata.
  2. Render the page to simplified wikitext: keep text-bearing templates (quotes, dialogue),
     turn tabbers into sub-headings, drop everything else (navboxes, galleries, refs, files, banners).
  3. Split into sections on headings, then strip each section to plain text and record link targets.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import mwparserfromhell as mw
from mwparserfromhell.nodes import Comment, Heading, Tag, Template, Text, Wikilink
from mwparserfromhell.wikicode import Wikicode

TAB_MARK = "\ue000TAB\ue000"  # private-use chars: survive the parser (NUL does not); level set when splitting
HEADING_RE = re.compile(r"^(={1,6})\s*(.+?)\s*\1\s*$")
DROP_LINK_PREFIXES = ("file:", "image:", "category:", "media:")
INTERLANG_RE = re.compile(r"^[a-z]{2,3}(-[a-z]+)?:", re.I)
DROP_TAGS = {"ref", "references", "gallery", "imagemap", "audio", "video", "math", "timeline"}
SKIP_SECTIONS = {"gallery", "references", "notes", "footnotes", "navigation", "external links",
                 "see also", "sources", "videos", "images"}


@dataclass
class Section:
    headings: list[str]  # e.g. ["Background", "NieR:Automata"]; [] for the lead
    text: str
    links_to: list[str] = field(default_factory=list)
    anchor: str | None = None  # nearest real heading (tabs have no anchor of their own)
    is_tab: bool = False  # the last heading is a tab name, e.g. an ending or route


@dataclass
class ParsedPage:
    page_id: int
    title: str
    revision_id: int
    url: str
    categories: list[str]
    aliases: list[str]
    infobox_type: str | None  # e.g. "character infobox", "ending infobox"
    infobox: dict[str, str]
    spoiler_banners: list[dict[str, str]]  # wiki's own {{Spoiler}} banners, input for M4
    sections: list[Section]
    links_to: list[str]


def template_name(t: Template) -> str:
    name = re.sub(r"<!--.*?-->", "", str(t.name), flags=re.S)
    return re.sub(r"[\s_]+", " ", name).strip().lower()


def _param(t: Template, *names: str | int) -> Wikicode | None:
    for n in names:
        if t.has(str(n)):
            v = t.get(str(n)).value
            if str(v).strip():
                return v
    return None


class Renderer:
    """Renders wikitext to simplified wikitext. Tracks templates it dropped, for review."""

    def __init__(self) -> None:
        self.page_title = ""  # for {{PAGENAME}}; set per page
        self.spoiler_banners: list[dict[str, str]] = []  # {{Spoiler|NA|Route=B}}; reset per page
        self.dropped: Counter[str] = Counter()
        self.handlers: dict[str, Callable[[Template], str]] = {
            "q": self._quote, "quote": self._quote, "bigquote": self._quote, "bigquotenoimg": self._quote,
            "pagename": lambda t: self.page_title,
            "character": lambda t: self.render(_param(t, "link", 1) or ""),
            "dialogue": self._dialogue,
            "w": lambda t: self.render(_param(t, 2, 1) or ""),
            "#tag:tabber": self._tag_tabber,
            "storytabs": self._story_tabs,
            "spoiler": self._spoiler,
        }

    def render(self, code: Wikicode | str) -> str:
        if isinstance(code, str):
            code = mw.parse(code)
        out: list[str] = []
        for node in code.nodes:
            if isinstance(node, Template):
                out.append(self._template(node))
            elif isinstance(node, Comment):
                continue
            elif isinstance(node, Heading):
                eq = "=" * node.level
                out.append(f"\n{eq} {self.render(node.title).strip()} {eq}\n")
            elif isinstance(node, Wikilink):
                out.append(self._link(node))
            elif isinstance(node, Tag):
                out.append(self._tag(node))
            else:
                out.append(str(node))
        return "".join(out)

    def _template(self, t: Template) -> str:
        name = template_name(t)
        if name in self.handlers:
            return self.handlers[name](t)
        self.dropped[name.split(":")[0] if name.startswith("#") else name] += 1
        return ""

    def _link(self, link: Wikilink) -> str:
        target = str(link.title).strip()
        low = target.lstrip(":").lower()
        if low.startswith(("w:", "wikipedia:")):  # interwiki: keep the text, no link target
            return self.render(link.text if link.text is not None else target.split(":")[-1])
        if target.startswith(":"):  # [[:Category:X|text]] is a visible link, not a category tag
            return self.render(link.text) if link.text is not None else target.lstrip(":").split(":", 1)[-1]
        if low.startswith(DROP_LINK_PREFIXES) or INTERLANG_RE.match(target):
            return ""
        text = self.render(link.text) if link.text is not None else target
        return f"[[{target}|{text}]]"

    def _tag(self, tag: Tag) -> str:
        name = str(tag.tag).strip().lower()
        if name in DROP_TAGS:
            return ""
        inner = self.render(tag.contents) if tag.contents is not None else ""
        if name in ("br", "p"):
            return "\n" + inner
        if name in ("tr", "li", "div"):
            return inner + "\n"
        if name in ("td", "th"):
            return inner.strip() + " | "
        return inner

    def _spoiler(self, t: Template) -> str:
        banner = {"game": str(_param(t, "Theme", 1) or "").strip()}
        for key in ("Route", "Ending"):
            if (v := _param(t, key)) is not None:
                banner[key.lower()] = plain_text(self.render(v))[0]
        self.spoiler_banners.append(banner)
        return ""

    def _quote(self, t: Template) -> str:
        q = self.render(_param(t, 1) or "").strip()
        src = _param(t, 2)
        return f'\n"{q}"' + (f" — {self.render(src).strip()}" if src else "") + "\n"

    def _dialogue(self, t: Template) -> str:
        return f"\n{self.render(_param(t, 2) or '').strip()}: {self.render(_param(t, 3) or '').strip()}\n"

    def _tag_tabber(self, t: Template) -> str:
        body = str(t)[2:-2].split("|", 1)[1] if "|" in str(t) else ""
        return self._tabs(re.split(r"\{\{\s*!\s*\}\}-\{\{\s*!\s*\}\}", body))

    def _tabs(self, parts: list[str]) -> str:
        out = []
        for part in parts:
            name, sep, content = part.partition("=")
            if not sep:
                continue
            out.append(f"\n{TAB_MARK}{self.render(name).strip()}\n{self.render(content)}\n")
        return "".join(out)

    def _story_tabs(self, t: Template) -> str:
        out = []
        for i in range(1, 11):
            text = _param(t, f"story{i}_text")
            if text is None:
                continue
            name = self.render(_param(t, f"story{i}_tabname") or f"Story {i}").strip()
            title = _param(t, f"story{i}_title")
            head = f"\n=== {self.render(title).strip()} ===\n" if title else ""
            out.append(f"\n{TAB_MARK}{name}\n{head}{self.render(text)}\n")
        return "".join(out)


def _tabber_to_marks(m: re.Match) -> str:
    out = []
    for part in re.split(r"\|-\|", m.group(1)):
        name, sep, content = part.partition("=")
        if sep:
            out.append(f"\n{TAB_MARK}{name.strip()}\n{content}\n")
    return "".join(out)


def preprocess(text: str) -> str:
    """Regex fixes for markup the parser can't handle reliably, applied before parsing."""
    # DISPLAYTITLE values can contain brackets that confuse the parser; it's display-only anyway.
    text = re.sub(r"\{\{\s*DISPLAYTITLE\s*:[^\n]*?\}\}", "", text, flags=re.I)
    text = re.sub(r"</?poem\b[^>]*>", "", text, flags=re.I)
    # <tabber> isn't always parsed as a tag (e.g. inside tables), so convert it to tab marks here.
    return re.sub(r"<tabber\b[^>]*>(.*?)</tabber>", _tabber_to_marks, text, flags=re.I | re.S)


def extract_infobox(code: Wikicode, renderer: Renderer) -> tuple[str | None, dict[str, str]]:
    """Remove the first infobox from `code`; return its type and non-empty fields as plain text."""
    for t in code.filter_templates(recursive=False):
        if "infobox" in template_name(t):
            fields = {}
            for p in t.params:
                key = str(p.name).strip().rstrip(":").strip()
                if key in ("image", "title") or key.endswith("_note"):
                    continue
                val = plain_text(renderer.render(p.value))[0].replace("\n", "; ")
                if val:
                    fields[key] = val
            code.remove(t)
            return template_name(t), fields
    return None, {}


def plain_text(wikitext: str) -> tuple[str, list[str]]:
    """Strip markup; return (text, raw link targets)."""
    code = mw.parse(wikitext)
    targets = [str(l.title).strip() for l in code.filter_wikilinks()]
    text = code.strip_code(normalize=True, collapse=True)
    text = re.sub(r"__[A-Z]+__", "", text)
    text = re.sub(r"</?[a-zA-Z][a-zA-Z0-9]*\b[^<>]*>", "", text)  # orphaned tags, e.g. <i> split across tabs
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"(\s*\|\s*)+\n", "\n", text)  # trailing table-cell separators
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), targets


@dataclass
class RawSection:
    headings: list[str]
    anchor: str | None
    is_tab: bool
    body: str


def split_sections(rendered: str) -> list[RawSection]:
    """Split rendered wikitext on headings. Tabs nest one level under the last real heading."""
    stack: list[tuple[int, str, bool]] = []  # (level, title, is_tab)
    base_level = 1
    sections = [RawSection([], None, False, "")]
    bodies: list[list[str]] = [[]]
    for line in rendered.split("\n"):
        m = HEADING_RE.match(line.strip())
        if m:
            level, title, tab = len(m.group(1)), plain_text(m.group(2))[0], False
            base_level = level
        elif line.startswith(TAB_MARK):
            level, title, tab = base_level + 1, line[len(TAB_MARK):].strip(), True
        else:
            bodies[-1].append(line)
            continue
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title, tab))
        anchor = next((t for _, t, is_tab in reversed(stack) if not is_tab), None)
        sections.append(RawSection([t for _, t, _ in stack], anchor, tab, ""))
        bodies.append([])
    for sec, body in zip(sections, bodies):
        sec.body = "\n".join(body)
    return sections


def parse_page(page, wiki, base_url: str, aliases: list[str], renderer: Renderer) -> ParsedPage:
    renderer.page_title = page.title
    renderer.spoiler_banners = []
    code = mw.parse(preprocess(page.text))
    infobox_type, infobox = extract_infobox(code, renderer)
    sections: list[Section] = []
    page_links: dict[str, None] = {}
    for raw in split_sections(renderer.render(code)):
        if any(h.lower() in SKIP_SECTIONS for h in raw.headings):
            continue
        text, raw_targets = plain_text(raw.body)
        if not text:
            continue
        links = list(dict.fromkeys(t for r in raw_targets if (t := wiki.resolve(r)) and t != page.title))
        page_links.update(dict.fromkeys(links))
        sections.append(Section(headings=raw.headings, text=text, links_to=links,
                                anchor=raw.anchor, is_tab=raw.is_tab))
    return ParsedPage(
        page_id=page.page_id,
        title=page.title,
        revision_id=page.revision_id,
        url=f"{base_url}/wiki/{page.title.replace(' ', '_')}",
        categories=page.categories,
        aliases=sorted(aliases),
        infobox_type=infobox_type,
        infobox=infobox,
        spoiler_banners=renderer.spoiler_banners,
        sections=sections,
        links_to=list(page_links),
    )


def to_json(p: ParsedPage) -> dict[str, Any]:
    return asdict(p)
