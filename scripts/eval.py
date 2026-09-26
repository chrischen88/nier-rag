"""Retrieval + answer eval over eval/questions.jsonl (SPEC §11). Writes reports/eval_<timestamp>.md
(and a .jsonl of per-question results).

  uv run python scripts/eval.py                                  # full run (calls the chat model)
  uv run python scripts/eval.py --retrieval-only                 # recall + retrieval leaks, no chat calls
  uv run python scripts/eval.py --embedding-model text-embedding-3-large   # needs that index built
  uv run python scripts/eval.py --set retrieval.mmr_lambda=0.7 --ids q001,q002
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, ConfigError, load_config, resolve_path  # noqa: E402
from src.evaluate import Result, load_questions, render_report, result_dict, score  # noqa: E402
from src.generate import Assistant  # noqa: E402
from src.index import IndexMismatchError, get_collection, open_client  # noqa: E402
from src.providers import ProviderError, get_embedder, get_llm, get_moderator  # noqa: E402
from src.providers.openai import estimate_cost  # noqa: E402
from src.retrieve import retrieve  # noqa: E402


def apply_override(cfg: dict, assignment: str) -> None:
    key, _, value = assignment.partition("=")
    *parents, leaf = key.split(".")
    node = cfg
    for p in parents:
        node = node[p]
    if leaf not in node:
        raise SystemExit(f"--set {key}: no such config key")
    node[leaf] = yaml.safe_load(value)


def run_one(q, assistant: Assistant, cfg: dict, retrieval_only: bool) -> Result:
    r = Result(q.id, q.question, q.user_level, q.expected_pages, q.should_refuse, q.leak_terms, [])
    try:
        if retrieval_only:
            passages = retrieve(assistant.col, assistant.embedder, q.question, q.user_level, cfg["retrieval"])
        else:
            ans = assistant.ask(q.question, q.user_level)
            passages = ans.passages
            r.answer, r.refused_before_llm, r.seconds = ans.text, ans.refused_before_llm, ans.seconds
            r.citations, r.invalid_citations = len(ans.citations), ans.invalid_citations
            r.guardrail = ans.guardrail
    except ProviderError as e:
        r.error = str(e)
        return r
    r.retrieved = [{"page_title": p.metadata["page_title"], "section_path": p.metadata["section_path"],
                    "spoiler_level": p.metadata["spoiler_level"], "similarity": round(p.similarity, 4)}
                   for p in passages]
    return r


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding-model", help="override embeddings.model")
    p.add_argument("--retrieval-only", action="store_true", help="skip the chat model")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. retrieval.min_similarity=0.4 (repeatable)")
    p.add_argument("--ids", help="comma-separated question ids to run")
    p.add_argument("--questions", default=str(ROOT / "eval/questions.jsonl"))
    args = p.parse_args()

    cfg = load_config()
    if args.embedding_model:
        cfg["embeddings"]["model"] = args.embedding_model
    for assignment in args.set:
        apply_override(cfg, assignment)

    questions = load_questions(Path(args.questions))
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q.id in wanted]

    try:
        embedder, llm = get_embedder(cfg), get_llm(cfg)
        col = get_collection(open_client(resolve_path(cfg, "chroma")), embedder)
    except (ConfigError, IndexMismatchError) as e:
        raise SystemExit(str(e))
    assistant = Assistant(cfg, col, embedder, llm, get_moderator(cfg))

    results = []
    for i, q in enumerate(questions, 1):
        r = run_one(q, assistant, cfg, args.retrieval_only)
        flags = [f for f, on in [("MISS", r.hit is False), ("RETRIEVAL-LEAK", r.retrieval_leaks),
                                 ("ANSWER-LEAK", r.leaked_terms), ("ERROR", r.error),
                                 ("REFUSAL-ERR", r.refused is not None and not r.is_trap
                                  and bool(r.refused) != r.should_refuse)] if on]
        print(f"[{i}/{len(questions)}] {q.id} L{q.user_level} {q.question[:60]:<60} {' '.join(flags)}")
        results.append(r)

    metrics = score(results)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    cost = estimate_cost(cfg, llm.usage, embedder.usage)
    meta = {
        "timestamp": stamp,
        "Mode": "retrieval only" if args.retrieval_only else "full",
        "Chat model": cfg["llm"]["model"],
        "Embedder": cfg["embeddings"]["model"],
        "Retrieval": ", ".join(f"{k}={v}" for k, v in cfg["retrieval"].items()),
        "Overrides": ", ".join(args.set) or "none",
        "Questions": f"{len(results)} ({sum(r.is_trap for r in results)} spoiler traps, "
                     f"{sum(r.should_refuse for r in results)} should refuse)",
        "Tokens": f"{llm.usage.prompt_tokens:,} prompt, {llm.usage.completion_tokens:,} completion, "
                  f"{embedder.usage.embedding_tokens:,} embedding",
        "Estimated cost": f"${cost:.4f} (embedding priced at pricing.embedding)",
    }
    out = ROOT / "reports" / f"eval_{stamp}"
    out.with_suffix(".md").write_text(render_report(results, metrics, meta))
    with open(out.with_suffix(".jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(result_dict(r)) + "\n")

    print()
    for m in metrics:
        mark = {True: "PASS", False: "FAIL", None: ""}[m.passed]
        value = "n/a" if m.value is None else f"{m.value:.3f}"
        print(f"  {m.name:<30} {value:>7}  {m.target:<8} n={m.n:<3} {mark}")
    print(f"\n{meta['Tokens']} · ~${cost:.4f}\nReport: {out.with_suffix('.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
