"""Spoiler-tag audit: find chunks that state a major twist but are tagged below the level that
reveals it. Exit code 1 if any are found. Fix them in data/spoiler_overrides.yaml."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path  # noqa: E402
from src.spoilers import SPOILER_PATTERNS  # noqa: E402


def main() -> int:
    cfg = load_config()
    chunks = [json.loads(line) for line in open(resolve_path(cfg, "chunks"))]
    found = 0
    for pattern, min_level in SPOILER_PATTERNS:
        hits = [c for c in chunks if pattern.search(c["text"].split("\n\n", 1)[-1])]
        low = [c for c in hits if c["spoiler_level"] < min_level]
        print(f"L{min_level}+ {len(low)}/{len(hits)} below  {pattern.pattern}")
        for c in low:
            body = c["text"].split("\n\n", 1)[-1]
            m = pattern.search(body)
            print(f"    L{c['spoiler_level']} ({c['spoiler_source']}) {c['section_path']}\n"
                  f"        …{body[max(0, m.start() - 70):m.end() + 50]!r}")
        found += len(low)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
