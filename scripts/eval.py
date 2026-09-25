"""Retrieval + answer eval over eval/questions.jsonl (SPEC §11, built in M6)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embedding-model", help="override embeddings.model")
    args = p.parse_args()
    cfg = load_config()
    if args.embedding_model:
        cfg["embeddings"]["model"] = args.embedding_model
    raise SystemExit("Not implemented yet (M6).")


if __name__ == "__main__":
    main()
