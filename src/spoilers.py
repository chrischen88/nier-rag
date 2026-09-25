"""Spoiler levels (SPEC §7.1) and the tagging pipeline (SPEC §7.2, built in M4)."""

from __future__ import annotations

import re

PROGRESS_LEVELS: dict[int, tuple[str, str]] = {
    0: ("Prologue", "Opening mission, basic premise"),
    1: ("Route A", "Playthrough 1 (2B), Ending A"),
    2: ("Route B", "Playthrough 2 (9S), Ending B"),
    3: ("Route C/D", "Playthrough 3 and Endings C/D"),
    4: ("Ending E", "Ending E"),
    5: ("Everything", "Side content, DLC, and lore revealed outside the game"),
}
MIN_LEVEL, MAX_LEVEL = 0, 5
DEFAULT_LEVEL = MAX_LEVEL  # when unsure, hide it


# Route/ending names a question can mention, and the level that unlocks them (SPEC §7.1).
_MARKERS = [
    (re.compile(r"\b(ending|branch)\s+e\b", re.I), 4),
    (re.compile(r"\b(route|ending|branch|playthrough)\s+(c|d|c\s*/\s*d|3)\b", re.I), 3),
    (re.compile(r"\b(route|ending|playthrough)\s+(b|2)\b", re.I), 2),
    (re.compile(r"\b(route|ending|playthrough)\s+(a|1)\b", re.I), 1),
]


def question_level(question: str) -> int | None:
    """Highest progress level a question explicitly names ("What happens in Ending E?" -> 4)."""
    levels = [lvl for pattern, lvl in _MARKERS if pattern.search(question)]
    return max(levels, default=None)


def tag_chunk(*args, **kwargs):
    raise NotImplementedError("M4: override -> heading -> category -> LLM -> default")
