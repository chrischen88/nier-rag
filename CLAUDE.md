# CLAUDE.md

YoRHa Archive: a spoiler-aware RAG assistant for NieR:Automata lore, built over a fixed snapshot of the NieR Fandom wiki. It uses OpenAI for embeddings and chat, with Chroma running locally. `SPEC.md` is the source of truth for design decisions, and section numbers in code comments (e.g. "SPEC §7.2") point into it. When behavior changes, update the SPEC's implementation notes too.

Milestones are listed in SPEC §13. M1–M6 are done. The next milestone is M7 (polish: README demo GIF, architecture diagram).

## Commands

```bash
uv sync                                              # deps (Python >=3.11, uv-managed; no package build)
uv run pytest                                        # all tests; no network or API key needed
uv run pytest tests/test_spoilers.py -k <name>       # single test
uv run python scripts/ingest.py                      # full pipeline: dump -> manifest -> parse -> chunk -> tag -> embed
uv run python scripts/ingest.py --reindex            # re-tag + re-embed data/chunks.jsonl (after an embedder change)
uv run python scripts/ingest.py --retag              # re-tag + update every index in place, no embedding (after overrides)
uv run python scripts/ingest.py --skip-llm --no-embed   # offline: rules only, stop after chunks.jsonl
uv run python scripts/query.py "Who is Pascal?" --level 1   # inspect retrieval
uv run python scripts/ask.py "Who is Pascal?" --level 1     # streamed answer + citations (--debug)
make serve                                           # chat UI (M5); `make help` lists shortcuts
uv run python scripts/audit_spoilers.py              # exit 1 if a twist is tagged below its reveal level
uv run python scripts/eval.py                        # SPEC §11 eval -> reports/eval_<ts>.md (~$0.01; --retrieval-only is ~free)
```

Ingest, query, and ask need `OPENAI_API_KEY` in `.env`. Ingest makes paid API calls, so ask before running it without `--skip-llm`/`--no-embed`.

## Architecture

- `src/ingest/`: `fetch` downloads the 7z XML dump, `dump` loads it, `select` builds `data/manifest.json` (SPEC §5.2), and `parse` renders wikitext with mwparserfromhell.
- `src/chunking.py`: section-level chunks (400 tokens max, 50 overlap, tiktoken `cl100k_base`) with a context header like `2B > Story > Route B`. The `Chunk` dataclass is the schema from SPEC §6.2.
- `src/spoilers.py`: progress levels 0–5 and the tagging pipeline. The first matching rule wins: manual override → heading rule → category/`{{Spoiler}}` banner → LLM classifier → default level 5.
- `src/index.py`: one Chroma collection per embedder (`chunks__openai_text-embedding-3-small`). `get_collection` fails if the index's embedder doesn't match the config.
- `src/retrieve.py`: filtered top-k, then MMR. `src/generate.py`'s `Assistant` handles `prepare` (question pre-check, similarity refusal) → `stream` → `finish` (citation resolution).
- `src/providers/`: `Embedder`/`LLM` protocols plus the `get_embedder`/`get_llm` factories. Only this package imports `openai`, and SDK errors are re-raised as `ProviderError`.
- `src/evaluate.py` (scoring, metrics, report) + `scripts/eval.py` (runner). `eval/questions.jsonl` is the SPEC §11 set; leak terms are a heuristic, so read trap answers by hand after tag changes.
- `app.py`: Streamlit UI over `Assistant`. It streams the answer, then redraws it with resolved citations. Chat history is not sent to the model, and turns asked at a higher level than the current slider are hidden.
- `scripts/*.py` add the repo root to `sys.path` and import `src.*` (hence the `# noqa: E402` lines).

## Invariants: don't break these

- **Every Chroma query goes through `spoiler_where()`**, which always includes `spoiler_level <= user_level`. This is the hard spoiler guarantee. The prompt is known to be weak on its own (SPEC §7.3), so never add a query path that bypasses the filter. `tests/test_spoiler_filter.py` and `tests/test_real_index.py` enforce this.
- When unsure, tag level 5. Don't lower defaults to "recover" content.
- The chat model is `gpt-4o-mini` only. `allowed_chat_models` in config is enforced by `get_llm()`. Model names live in `config.yaml`, never in code.
- The API key is read only from the `OPENAI_API_KEY` env var. Never put it in config and never log it.
- Chunk context headers use `chunking.display_names`, not wiki redirects, because some aliases are spoilers (e.g. `2E`). The same applies to `infobox_text_fields`, which is an allow-list: fields like `aka`/`status` leak.
- Every answer must cite wiki sources (CC BY-SA attribution).

## Data and spoiler-tag workflow

- `data/` and `reports/` are gitignored except `data/manifest.json` (hand-editable; `--rebuild-manifest` overwrites it), `data/spoiler_overrides.yaml`, and `reports/spoiler_tags.csv`.
- To fix a tag: add an override to `data/spoiler_overrides.yaml`, run `ingest.py --retag`, then run `audit_spoilers.py`. Key overrides by section path where possible. A lead chunk's section path equals the page title, and a title key overrides the *whole* page (including any L5 sections), so key leads by `chunk_id`. Don't use `--reindex` for tag changes: it rebuilds only the configured embedder's index and leaves the others stale. LLM classifier results are cached in `data/spoiler_llm_cache.jsonl`, keyed on `PROMPT_VERSION`. Bump that constant when you change the classifier prompt.
- Changing `embeddings.model` requires `--reindex`. Changing the chat model does not.
- Tunable thresholds (`retrieval.min_similarity`, `mmr_lambda`) are in `config.yaml`, along with notes on how they were chosen.

## Testing conventions

- Tests use `FakeEmbedder` and the `openai_key` fixture from `tests/conftest.py`. Keep unit tests offline.
- `tests/test_real_index.py` runs against the built index and skips when `data/chroma` is missing.
- `hypothesis` is available for property tests.
- `tests/test_app.py` drives `app.py` through Streamlit's `AppTest`, monkeypatching `src.providers.get_*`, `src.index.*`, and `src.generate.retrieve`.
