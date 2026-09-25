"""Chunk schema (SPEC §6.2) and section chunker (SPEC §6.1)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any
from urllib.parse import quote

import tiktoken

from src.spoilers import DEFAULT_LEVEL


class ContentType(str, Enum):
    CHARACTER = "character"
    STORY = "story"
    LOCATION = "location"
    MACHINE = "machine"
    WEAPON_STORY = "weapon_story"
    ITEM = "item"
    QUEST = "quest"
    TRIVIA = "trivia"
    SPECULATION = "speculation"
    META = "meta"


class SpoilerSource(str, Enum):
    HEADING_RULE = "heading_rule"
    CATEGORY_RULE = "category_rule"
    LLM = "llm"
    MANUAL = "manual"
    DEFAULT = "default"


@dataclass
class Chunk:
    chunk_id: str  # "{page_id}:{section_idx}:{part}"
    page_title: str
    section_path: str  # e.g. "9S > Story > Route C"
    url: str
    text: str  # cleaned text, starting with the context header
    content_type: ContentType
    spoiler_level: int
    spoiler_source: SpoilerSource
    revision_id: int
    categories: list[str] = field(default_factory=list)
    is_speculation: bool = False
    infobox: dict[str, Any] | None = None  # first chunk of a page only
    links_to: list[str] = field(default_factory=list)
    spoiler_banners: list[dict[str, str]] = field(default_factory=list)  # wiki {{Spoiler}} banners on this
    # section or an enclosing one; input for the spoiler tagger's category rule

    def __post_init__(self) -> None:
        self.content_type = ContentType(self.content_type)
        self.spoiler_source = SpoilerSource(self.spoiler_source)
        if not 0 <= self.spoiler_level <= 5:
            raise ValueError(f"{self.chunk_id}: spoiler_level {self.spoiler_level} not in 0..5")

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["content_type"] = self.content_type.value
        d["spoiler_source"] = self.spoiler_source.value
        return d

    def chroma_metadata(self) -> dict[str, str | int | bool]:
        """Scalar fields only; lists and dicts stay in data/chunks.jsonl."""
        return {
            "page_title": self.page_title,
            "section_path": self.section_path,
            "url": self.url,
            "content_type": self.content_type.value,
            "spoiler_level": self.spoiler_level,
            "spoiler_source": self.spoiler_source.value,
            "is_speculation": self.is_speculation,
            "revision_id": self.revision_id,
        }


# --- token helpers ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")  # tokenizer of text-embedding-3-*


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


SENTENCE_RE = re.compile(r"(?<=[.!?…。])[\"'”’)]*\s+")
SPECULATION_RE = re.compile(r"\b(trivia|theor(y|ies)|speculation)\b", re.I)
# Headings that mark route/ending/chapter boundaries; these sections are never merged away,
# so the spoiler tagger (M4) can still see them.
ROUTE_MARKER_RE = re.compile(r"\b(route|ending|endings|chapter|playthrough)\b|^[A-Z]$", re.I)

INFOBOX_TYPES = {
    "item infobox": ContentType.ITEM,
    "archiveinfoboxnierautomata": ContentType.ITEM,
    "infobox - pod program": ContentType.ITEM,
    "infobox - unit data": ContentType.MACHINE,
    "unit data infobox": ContentType.MACHINE,
    "quest infobox (automata)": ContentType.QUEST,
    "infobox - weapon (automata)": ContentType.WEAPON_STORY,
    "ending infobox": ContentType.STORY,
    "chapter infobox": ContentType.STORY,
    "lore infobox": ContentType.STORY,
    "organization infobox": ContentType.STORY,
    "character infobox": ContentType.CHARACTER,
    "dlc infobox": ContentType.META,
    "cd infobox": ContentType.META,
    "book infobox": ContentType.META,
    "game infobox": ContentType.META,
}
CATEGORY_TYPES = [  # fallback when there's no known infobox; first match wins
    (re.compile(r"weapon", re.I), ContentType.WEAPON_STORY),
    (re.compile(r"quest", re.I), ContentType.QUEST),
    (re.compile(r"machine|units|bosses", re.I), ContentType.MACHINE),
    (re.compile(r"location", re.I), ContentType.LOCATION),
    (re.compile(r"character|android", re.I), ContentType.CHARACTER),
    (re.compile(r"item|material|archive", re.I), ContentType.ITEM),
]


def page_content_type(page: dict[str, Any]) -> ContentType:
    if (t := INFOBOX_TYPES.get(page.get("infobox_type") or "")) is not None:
        return t
    for pattern, ctype in CATEGORY_TYPES:
        if any(pattern.search(c) for c in page["categories"]):
            return ctype
    if page["infobox"].keys() & {"race", "va_EN", "va_JP"}:
        return ContentType.CHARACTER
    return ContentType.STORY


def section_content_type(headings: list[str], page_type: ContentType) -> ContentType:
    for h in headings:
        if re.search(r"\btrivia\b", h, re.I):
            return ContentType.TRIVIA
        if re.search(r"\b(theor(y|ies)|speculation)\b", h, re.I):
            return ContentType.SPECULATION
    return page_type


# --- merging and splitting -------------------------------------------------------------------

def _mergeable(sec: dict[str, Any]) -> bool:
    return bool(sec["headings"]) and not sec.get("is_tab") and not any(
        ROUTE_MARKER_RE.search(h) or SPECULATION_RE.search(h) for h in sec["headings"])


def merge_short_sections(sections: list[dict[str, Any]], min_tokens: int) -> list[dict[str, Any]]:
    """Fold sections under `min_tokens` into their parent section (or, if the parent has no text
    of its own, into the previous sibling). Route/ending tabs and speculation are never merged."""
    kept: list[dict[str, Any]] = []
    for sec in sections:
        sec = {**sec, "links_to": list(sec["links_to"]), "spoiler_banners": list(sec.get("spoiler_banners", []))}
        if _mergeable(sec) and count_tokens(sec["text"]) < min_tokens:
            parent = sec["headings"][:-1]
            target = next((k for k in reversed(kept) if k["headings"] == parent), None)
            if target is None and kept and kept[-1]["headings"][:-1] == parent and _mergeable(kept[-1]):
                target = kept[-1]
            if target is not None:
                target["text"] += f"\n\n{sec['headings'][-1]}: {sec['text']}"
                target["links_to"] += [l for l in sec["links_to"] if l not in target["links_to"]]
                target["spoiler_banners"] += [b for b in sec["spoiler_banners"] if b not in target["spoiler_banners"]]
                continue
        kept.append(sec)
    return kept


def _units(text: str, max_tokens: int) -> list[str]:
    """Paragraphs, with oversized paragraphs broken into sentences (and oversized sentences
    into token windows)."""
    out: list[str] = []
    for para in (p.strip() for p in text.split("\n") if p.strip()):
        if count_tokens(para) <= max_tokens:
            out.append(para)
            continue
        for sent in SENTENCE_RE.split(para):
            if count_tokens(sent) <= max_tokens:
                out.append(sent)
            else:
                ids = _encoding().encode(sent)
                out += [_encoding().decode(ids[i:i + max_tokens]) for i in range(0, len(ids), max_tokens)]
    return out


def _tail(text: str, overlap_tokens: int) -> str:
    """Last whole sentences of `text` totalling at most `overlap_tokens`."""
    tail: list[str] = []
    for sent in reversed(SENTENCE_RE.split(text)):
        if count_tokens(" ".join([sent, *tail])) > overlap_tokens:
            break
        tail.insert(0, sent)
    return " ".join(tail)


def split_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    for unit in _units(text, max_tokens):
        if current and count_tokens("\n".join([*current, unit])) > max_tokens:
            parts.append("\n".join(current))
            overlap = _tail(current[-1], overlap_tokens)
            current = [overlap] if overlap else []
            if current and count_tokens("\n".join([*current, unit])) > max_tokens:
                current = []
        current.append(unit)
    if current:
        parts.append("\n".join(current))
    return parts


# --- page -> chunks --------------------------------------------------------------------------

def section_url(page_url: str, anchor: str | None) -> str:
    if not anchor:
        return page_url
    return f"{page_url}#{quote(anchor.replace(' ', '_'), safe=':()!,.-_~')}"


def infobox_summary(infobox: dict[str, str], fields: dict[str, str]) -> str:
    """Allow-listed infobox fields as text, e.g. "English voice: Kira Buckland"."""
    return "\n".join(f"{label}: {infobox[key]}" for key, label in fields.items() if infobox.get(key))


def chunk_page(page: dict[str, Any], chunk_cfg: dict[str, Any]) -> list[Chunk]:
    display = chunk_cfg.get("display_names", {}).get(page["title"], page["title"])
    page_type = page_content_type(page)
    sections = [dict(s) for s in page["sections"]]
    summary = infobox_summary(page["infobox"], chunk_cfg.get("infobox_text_fields", {}))
    if summary:  # the lead carries the page's key facts; create one if the page has no lead text
        if sections and not sections[0]["headings"]:
            sections[0]["text"] = f"{sections[0]['text']}\n\n{summary}"
        else:
            sections.insert(0, {"headings": [], "text": summary, "links_to": [], "anchor": None, "is_tab": False})
    sections = merge_short_sections(sections, chunk_cfg["min_tokens"])
    chunks: list[Chunk] = []
    for idx, sec in enumerate(sections):
        banners = [b for other in sections if other["headings"] == sec["headings"][:len(other["headings"])]
                   for b in other.get("spoiler_banners", [])]  # this section's and its ancestors'
        banners = [b for i, b in enumerate(banners) if b not in banners[:i]]
        section_path = " > ".join([display, *sec["headings"]])
        ctype = section_content_type(sec["headings"], page_type)
        for part, body in enumerate(split_text(sec["text"], chunk_cfg["max_tokens"], chunk_cfg["overlap_tokens"])):
            chunks.append(Chunk(
                chunk_id=f"{page['page_id']}:{idx}:{part}",
                page_title=page["title"],
                section_path=section_path,
                url=section_url(page["url"], sec.get("anchor")),
                text=f"{section_path}\n\n{body}",
                content_type=ctype,
                spoiler_level=DEFAULT_LEVEL,  # M4 replaces this with the tagging pipeline
                spoiler_source=SpoilerSource.DEFAULT,
                revision_id=page["revision_id"],
                categories=page["categories"],
                is_speculation=ctype in (ContentType.TRIVIA, ContentType.SPECULATION),
                infobox=(page["infobox"] or None) if not chunks else None,
                links_to=sec["links_to"],
                spoiler_banners=banners,
            ))
    return chunks


def validate_chunks(chunks: list[Chunk], max_embed_tokens: int = 8191) -> None:
    """Schema checks beyond Chunk.__post_init__. Raises ValueError on the first problem."""
    seen: set[str] = set()
    for c in chunks:
        if c.chunk_id in seen:
            raise ValueError(f"duplicate chunk_id {c.chunk_id}")
        seen.add(c.chunk_id)
        if not re.fullmatch(r"\d+:\d+:\d+", c.chunk_id):
            raise ValueError(f"bad chunk_id {c.chunk_id!r}")
        if not c.text.startswith(c.section_path) or not c.text[len(c.section_path):].strip():
            raise ValueError(f"{c.chunk_id}: text must be the context header plus a non-empty body")
        if not c.url.startswith("https://"):
            raise ValueError(f"{c.chunk_id}: bad url {c.url!r}")
        if count_tokens(c.text) > max_embed_tokens:
            raise ValueError(f"{c.chunk_id}: too long to embed")
