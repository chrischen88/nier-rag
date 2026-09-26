"""CLI Q&A: stream an answer with citations at a given progress level."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path  # noqa: E402
from src.generate import Assistant  # noqa: E402
from src.index import get_collection, open_client  # noqa: E402
from src.providers import get_embedder, get_llm, get_moderator  # noqa: E402
from src.providers.openai import estimate_cost  # noqa: E402
from src.spoilers import MAX_LEVEL, PROGRESS_LEVELS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("question")
    p.add_argument("--level", type=int, default=MAX_LEVEL, choices=range(MAX_LEVEL + 1),
                   help="progress level 0-5 (default 5, everything)")
    p.add_argument("--hide-speculation", action="store_true")
    p.add_argument("--debug", action="store_true", help="also print retrieved passages and scores")
    args = p.parse_args()

    cfg = load_config()
    embedder, llm = get_embedder(cfg), get_llm(cfg)
    assistant = Assistant(cfg, get_collection(open_client(resolve_path(cfg, "chroma")), embedder), embedder, llm,
                          get_moderator(cfg))

    print(f"[{PROGRESS_LEVELS[args.level][0]}] {args.question}\n")
    plan = assistant.prepare(args.question, args.level, hide_speculation=args.hide_speculation)
    parts = []
    for token in assistant.stream(plan):
        parts.append(token)
        print(token, end="", flush=True)
    answer = assistant.finish(plan, "".join(parts))
    print("\n")
    for c in answer.citations:
        print(c)
    if answer.invalid_citations:
        print(f"(removed invalid citations: {answer.invalid_citations})")
    cost = estimate_cost(cfg, llm.usage, embedder.usage)
    print(f"\n{answer.seconds:.1f}s · {llm.usage.prompt_tokens} in / {llm.usage.completion_tokens} out tokens"
          f" · ~${cost:.5f}" + (" · refused before calling the LLM" if answer.refused_before_llm else ""))
    if args.debug:
        print("\nRetrieved:")
        for i, r in enumerate(answer.passages, 1):
            m = r.metadata
            print(f"  [{i}] {r.similarity:.3f} L{m['spoiler_level']} {m['section_path']}")


if __name__ == "__main__":
    main()
