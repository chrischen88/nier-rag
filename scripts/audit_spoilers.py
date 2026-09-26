"""Spoiler-tag audit: find chunks that state a major twist but are tagged below the level that
reveals it. Exit code 1 if any are found. Fix them in data/spoiler_overrides.yaml."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path  # noqa: E402

# (pattern on chunk body, level at which it stops being a spoiler). Expect some false positives.
TWISTS = [
    (r"human(s|ity)?\b[^.]{0,80}\bextinct|extinct[^.]{0,60}\bhuman", 2),
    (r"\b2E\b|\bType E\b", 3),
    (r"black box[^.]{0,120}(machine core|core of a machine|same (core|material))", 3),
    (r"YoRHa (were|was) (designed|planned|meant) to be (destroyed|disposed)|planned disposal|YoRHa Disposal", 3),
    (r"\bEnding E\b", 4),
    (r"\bRed Girls?\b", 2),  # the Red Girls first appear at the end of Route B
    (r"\bChapter 1[1-7]\b", 3),  # Routes A/B end at Chapter 10; later chapters are Route C/D
    (r"Resource Recovery Units?|Access (Release )?Keys?\b|Tower sub-?(unit|node)s?", 3),  # Route C Tower entry
]


def main() -> int:
    cfg = load_config()
    chunks = [json.loads(line) for line in open(resolve_path(cfg, "chunks"))]
    found = 0
    for pattern, min_level in TWISTS:
        hits = [c for c in chunks if re.search(pattern, c["text"].split("\n\n", 1)[-1], re.I)]
        low = [c for c in hits if c["spoiler_level"] < min_level]
        print(f"L{min_level}+ {len(low)}/{len(hits)} below  {pattern}")
        for c in low:
            body = c["text"].split("\n\n", 1)[-1]
            m = re.search(pattern, body, re.I)
            print(f"    L{c['spoiler_level']} ({c['spoiler_source']}) {c['section_path']}\n"
                  f"        …{body[max(0, m.start() - 70):m.end() + 50]!r}")
        found += len(low)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
