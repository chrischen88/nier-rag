"""CLI retrieval check: show the chunks a question retrieves at a given progress level."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path  # noqa: E402
from src.index import get_collection, open_client  # noqa: E402
from src.providers import get_embedder  # noqa: E402
from src.retrieve import retrieve  # noqa: E402
from src.spoilers import MAX_LEVEL, PROGRESS_LEVELS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("question")
    p.add_argument("--level", type=int, default=MAX_LEVEL, choices=range(MAX_LEVEL + 1),
                   help="progress level 0-5 (default 5, everything)")
    p.add_argument("--type", action="append", dest="content_types", help="content_type filter (repeatable)")
    p.add_argument("--hide-speculation", action="store_true")
    p.add_argument("--chars", type=int, default=300, help="characters of each chunk to print")
    args = p.parse_args()

    cfg = load_config()
    embedder = get_embedder(cfg)
    col = get_collection(open_client(resolve_path(cfg, "chroma")), embedder)
    results = retrieve(col, embedder, args.question, args.level, cfg["retrieval"],
                       content_types=args.content_types, hide_speculation=args.hide_speculation)
    print(f"Q: {args.question}  [level {args.level}: {PROGRESS_LEVELS[args.level][0]}]\n")
    for i, r in enumerate(results, 1):
        m = r.metadata
        body = r.text.split("\n\n", 1)[-1].replace("\n", " ")
        print(f"[{i}] {r.similarity:.3f}  {m['section_path']}  (L{m['spoiler_level']}, {m['content_type']})")
        print(f"    {m['url']}")
        print(f"    {body[:args.chars]}{'…' if len(body) > args.chars else ''}\n")


if __name__ == "__main__":
    main()
