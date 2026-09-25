"""Spoiler levels (SPEC §7.1) and the tagging pipeline (SPEC §7.2).

Rules run in order and the first that matches wins:
  1. manual override (data/spoiler_overrides.yaml: chunk_id, section path, or page title)
  2. heading rule (route/ending markers and config heading patterns in the section path)
  3. category rule (wiki {{Spoiler}} banners on the section, then config category patterns)
  4. LLM classifier (accepted at confidence >= threshold)
  5. default: level 5
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from src.chunking import Chunk
    from src.providers.base import LLM

PROGRESS_LEVELS: dict[int, tuple[str, str]] = {
    0: ("Prologue", "Opening mission, basic premise"),
    1: ("Route A", "Playthrough 1 (2B), Ending A"),
    2: ("Route B", "Playthrough 2 (9S), Ending B"),
    3: ("Route C/D", "Playthrough 3 and Endings C/D"),
    4: ("Ending E", "Ending E"),
    5: ("Everything", "DLC, and lore revealed outside the game (anime, stage plays, novels, other games)"),
}
MIN_LEVEL, MAX_LEVEL = 0, 5
DEFAULT_LEVEL = MAX_LEVEL  # when unsure, hide it

LETTER_LEVELS = {"A": 1, "B": 2, "C": 3, "D": 3, "E": 4}
AUTOMATA_GAMES = {"na", "nier:automata", "nier: automata", "automata"}

# Route/ending names a question can mention, and the level that unlocks them (SPEC §7.1).
_MARKERS = [
    (re.compile(r"\b(ending|branch)\s+e\b", re.I), 4),
    (re.compile(r"\b(route|ending|branch|playthrough)\s+(c|d|c\s*/\s*d|3)\b", re.I), 3),
    (re.compile(r"\b(route|ending|playthrough)\s+(b|2)\b", re.I), 2),
    (re.compile(r"\b(route|ending|playthrough)\s+(a|1)\b", re.I), 1),
]
_HEADING_MARKER = re.compile(r"\b(?:route|ending|branch)s?\s+([A-E](?:\s*(?:/|,|and|&)\s*[A-E])*)\b", re.I)


def question_level(question: str) -> int | None:
    """Highest progress level a question explicitly names ("What happens in Ending E?" -> 4)."""
    levels = [lvl for pattern, lvl in _MARKERS if pattern.search(question)]
    return max(levels, default=None)


def _letters_level(letters: str) -> int | None:
    levels = [LETTER_LEVELS[c] for c in re.findall(r"[A-E]", letters.upper())]
    return max(levels, default=None)


def heading_marker_level(headings: list[str]) -> int | None:
    """Route/ending markers in a section path: "Route B", "Ending E", an "Endings > C" tab, or an
    ending page's title ("The (E)nd of YoRHa")."""
    levels = []
    if headings and (m := re.search(r"\(([A-E])\)", headings[0])):  # ending pages: "The (E)nd of YoRHa"
        levels.append(LETTER_LEVELS[m.group(1)])
    for i, h in enumerate(headings):
        for m in _HEADING_MARKER.finditer(h):
            levels.append(_letters_level(m.group(1)))
        if re.fullmatch(r"[A-E]", h.strip()) and i > 0 and re.search(r"\bendings?\b", headings[i - 1], re.I):
            levels.append(LETTER_LEVELS[h.strip()])
    levels = [lvl for lvl in levels if lvl is not None]
    return max(levels, default=None)


def banner_level(banners: list[dict[str, str]]) -> int | None:
    """Level from wiki {{Spoiler|NA|Route=C/D}} banners. Banners for other games or with no route are ignored."""
    levels = []
    for b in banners:
        if b.get("game", "").strip().lower() not in AUTOMATA_GAMES:
            continue
        lvl = _letters_level(" ".join(b.get(k, "") for k in ("route", "ending")))
        if lvl is not None:
            levels.append(lvl)
    return max(levels, default=None)


# --- LLM classifier --------------------------------------------------------------------------

PROMPT_VERSION = "v3"
CLASSIFIER_PROMPT = """You tag NieR:Automata wiki passages with a spoiler level for a spoiler-aware \
lore app. A player at level N may only see passages tagged N or lower.

Levels:
{levels}

Return the LOWEST level at which the passage spoils nothing for the player. By then the player \
must (a) have met everything the passage names (characters, places, enemies, items, events) and \
(b) have seen every fact or reveal it states.
- Level 0 is only for the opening mission: 2B, 9S, Pods, the Abandoned Factory, the Goliath-class \
Engels, the Bunker, Commander White, Operator 6O, and the basic premise (YoRHa androids fight the \
aliens' machine lifeforms on behalf of humans who fled to the Moon). Anything first met after the \
prologue is level 1 or higher, even if the passage contains no plot.
- In-game text and side content (weapon stories, archives, item and enemy descriptions, shops, \
side quests, fishing) take the level at which they first become available in play, even when they \
allude to other NieR games. Gameplay-only content (stats, controls) works the same way.
- Known reveal points, from the wiki's own spoiler banners: humanity's extinction, and the Council \
of Humanity on the Moon being a fabrication (Project YoRHa), are Route B (level 2) at the earliest. \
Black boxes being made from machine cores, 2B's true designation 2E, and YoRHa's planned disposal \
are Route C/D (level 3).
- Level 5 is only for DLC content (the 3C3C1D119440927 DLC, its colosseums and items) and for \
passages whose subject is outside NieR:Automata (other NieR or Drakengard games, the anime, stage \
plays, novels, concerts, crossovers).
- Judge every sentence: one late detail makes the whole passage late. Use your knowledge of \
NieR:Automata to judge when things are revealed. If unsure between two levels, choose the higher \
one and lower your confidence.

Reply with JSON only: {{"level": <0-5>, "confidence": <0.0-1.0>, "reason": "<at most 15 words>"}}"""


@dataclass
class TagResult:
    level: int
    source: str  # manual | heading_rule | category_rule | llm | default
    confidence: float | None = None
    llm_level: int | None = None
    reason: str = ""


class LLMClassifier:
    def __init__(self, llm: LLM, threshold: float, cache_path: Path):
        self.llm, self.threshold, self.cache_path = llm, threshold, cache_path
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row["result"]
        self._lock = threading.Lock()
        self.calls = 0

    def _key(self, chunk: Chunk) -> str:
        raw = json.dumps([PROMPT_VERSION, self.llm.name, chunk.text, chunk.categories])
        return hashlib.sha256(raw.encode()).hexdigest()

    def classify(self, chunk: Chunk) -> dict:
        key = self._key(chunk)
        if key in self.cache:
            return self.cache[key]
        levels = "\n".join(f"{i}: {label} — {desc}" for i, (label, desc) in PROGRESS_LEVELS.items())
        messages = [
            {"role": "system", "content": CLASSIFIER_PROMPT.format(levels=levels)},
            {"role": "user", "content": f"Page categories: {', '.join(chunk.categories) or 'none'}\n\n"
                                        f"Passage:\n{chunk.text}"},
        ]
        try:
            out = json.loads(self.llm.generate(messages, json_mode=True))
            result = {"level": int(out["level"]), "confidence": float(out["confidence"]),
                      "reason": str(out.get("reason", ""))[:200]}
            if not MIN_LEVEL <= result["level"] <= MAX_LEVEL:
                raise ValueError(f"level {result['level']} out of range")
        except (ValueError, KeyError, TypeError) as e:  # malformed reply: never cache, fall to default
            return {"level": None, "confidence": 0.0, "reason": f"unparseable reply: {e}"}
        with self._lock:
            self.calls += 1
            self.cache[key] = result
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "result": result}) + "\n")
        return result


# --- tagger ----------------------------------------------------------------------------------

def load_overrides(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    for key, level in data.items():
        if not isinstance(level, int) or not MIN_LEVEL <= level <= MAX_LEVEL:
            raise ValueError(f"{path}: override for {key!r} must be an int 0..5, got {level!r}")
    return {str(k): v for k, v in data.items()}


class SpoilerTagger:
    def __init__(self, spoiler_cfg: dict[str, Any], overrides: dict[str, int],
                 classifier: LLMClassifier | None):
        self.overrides = overrides
        self.heading_rules = [(re.compile(r["pattern"]), r["level"]) for r in spoiler_cfg.get("heading_rules", [])]
        self.category_rules = [(re.compile(r["pattern"]), r["level"]) for r in spoiler_cfg.get("category_rules", [])]
        self.classifier = classifier

    def rule_tag(self, chunk: Chunk) -> TagResult | None:
        """Rules 1-3. None means no rule matched."""
        for key in (chunk.chunk_id, chunk.section_path, chunk.page_title):
            if key in self.overrides:
                return TagResult(self.overrides[key], "manual", reason=f"override: {key}")

        headings = chunk.section_path.split(" > ")
        levels = [lvl for pattern, lvl in self.heading_rules for h in headings if pattern.search(h)]
        if (marker := heading_marker_level(headings)) is not None:
            levels.append(marker)
        if levels:
            return TagResult(max(levels), "heading_rule")

        levels = [lvl for pattern, lvl in self.category_rules for c in chunk.categories if pattern.search(c)]
        if (banner := banner_level(chunk.spoiler_banners)) is not None:
            levels.append(banner)
        if levels:
            return TagResult(max(levels), "category_rule")
        return None

    def tag(self, chunk: Chunk) -> TagResult:
        if (result := self.rule_tag(chunk)) is not None:
            return result
        if self.classifier is None:
            return TagResult(DEFAULT_LEVEL, "default", reason="LLM classifier skipped")
        out = self.classifier.classify(chunk)
        if out["level"] is not None and out["confidence"] >= self.classifier.threshold:
            return TagResult(out["level"], "llm", out["confidence"], out["level"], out["reason"])
        return TagResult(DEFAULT_LEVEL, "default", out["confidence"], out["level"], out["reason"])

    def tag_all(self, chunks: list[Chunk], workers: int = 8) -> list[TagResult]:
        """Tag chunks in place (spoiler_level, spoiler_source) and return the results."""
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(self.tag, chunks))
        from src.chunking import SpoilerSource  # runtime import: chunking imports this module

        for chunk, r in zip(chunks, results):
            chunk.spoiler_level = r.level
            chunk.spoiler_source = SpoilerSource(r.source)
        return results


def write_report(path: Path, chunks: list[Chunk], results: list[TagResult]) -> int:
    """CSV of LLM- and default-tagged chunks for manual review (SPEC §7.2). Returns row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "page_title", "section_path", "level", "source", "llm_level",
                    "confidence", "reason", "url", "preview"])
        for c, r in zip(chunks, results):
            if r.source not in ("llm", "default"):
                continue
            preview = c.text.split("\n\n", 1)[-1].replace("\n", " ")[:300]
            w.writerow([c.chunk_id, c.page_title, c.section_path, r.level, r.source,
                        "" if r.llm_level is None else r.llm_level,
                        "" if r.confidence is None else f"{r.confidence:.2f}", r.reason, c.url, preview])
            rows += 1
    return rows
