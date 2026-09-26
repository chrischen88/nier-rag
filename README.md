# YoRHa Archive

Spoiler-aware NieR:Automata lore assistant: RAG over the NieR Fandom wiki, using OpenAI for embeddings and chat and a local Chroma index. You set how far you've played, and the app never retrieves content past that point. See [SPEC.md](SPEC.md).

## Setup

```bash
uv sync                 # Python deps + dev tools
cp .env.example .env    # then add OPENAI_API_KEY
uv run pytest           # unit tests, including the spoiler-filter invariant
```

Models, paths, and thresholds live in `config.yaml`.

## Run

```bash
uv run python scripts/ingest.py     # dump -> manifest -> parse -> chunk -> spoiler tags -> embed into Chroma
uv run python scripts/ingest.py --reindex    # re-tag + re-embed data/chunks.jsonl (after changing the embedder)
uv run python scripts/ingest.py --retag      # re-tag only and update every index in place (after editing overrides)
uv run python scripts/query.py "Who is Pascal?" --level 1   # inspect retrieval at a progress level
uv run python scripts/ask.py "Who is Pascal?" --level 1     # streamed answer with citations (--debug shows passages)
uv run python scripts/audit_spoilers.py   # flag chunks that state a twist below its reveal level
```

Spoiler tags: review `reports/spoiler_tags.csv`, fix tags in `data/spoiler_overrides.yaml`, then run
`make retag` (classifier results are cached and nothing is re-embedded, so it's free) and `make audit`.

```bash
make serve                          # chat UI (uv run streamlit run app.py); PORT=9000 to change port
uv run python scripts/eval.py       # eval -> reports/eval_<timestamp>.md (--retrieval-only, --embedding-model, --set)
```

## Guardrails

- **Spoilers:**
  - Retrieval never returns chunks above the player's level (the hard guarantee).
  - Questions naming a later route or ending are refused before any model call.
  - Streamed answers are checked against known twists and blocked before a twist is shown, which catches the model's own knowledge of the plot.
- **Scope:**
  - Gameplay questions and questions with no relevant wiki passage are refused before any model call.
  - The prompt answers only from the cited passages and declines anything that isn't lore.
- **Abuse:**
  - Questions are limited to 500 characters and wrapped in delimiters the user can't break out of.
  - OpenAI's free moderation endpoint screens questions for harassment, hate, sexual, violent-wrongdoing, and self-harm content. Self-harm gets a reply with crisis resources.
  - Plain "violence" is ignored, because the game is violent.
- **Cost:** see the daily cap and password below.

Settings live under `guardrails` in `config.yaml`. `eval/adversarial.jsonl` tests them:

```bash
uv run python scripts/eval.py --questions eval/adversarial.jsonl
```

## Deploy (Fly.io)

The image bakes in the local index (`data/chroma`), so build it first with `make ingest`. After changing
spoiler tags, run `make retag` before deploying. Deploys make no OpenAI calls.

```bash
fly apps create yorha-archive                 # or pick another name and update `app` in fly.toml
fly secrets set OPENAI_API_KEY=sk-...
fly secrets set APP_PASSWORD=...              # optional: ask visitors for a password
make deploy                                   # fly deploy --ha=false
```

A public URL spends your OpenAI credits, so:

- `app.daily_question_limit` in `config.yaml` (300 by default, about $0.06 a day) caps questions per machine per day.
- Set a monthly spending limit on the OpenAI account as a backstop.

The machine stops when idle and cold-starts in about 10 seconds on the next visit.

## Evaluation

`eval/questions.jsonl` has 60 hand-written questions: 32 factual (7 needing several pages), 9 that should be
refused (gameplay, off-topic, or naming a later route), and 19 spoiler traps asked at a low progress level with
terms that must not appear in the answer. `scripts/eval.py` scores them against the SPEC §11 targets.

Results on 2026-09-26, `gpt-4o-mini` for chat, same index and tags:

| Metric | Target | `text-embedding-3-small` | `text-embedding-3-large` |
|---|---|---|---|
| Recall@6 | ≥ 0.80 | 0.97 | 1.00 |
| Retrieval leak rate | 0 | 0 | 0 |
| Answer leak rate | ≤ 0.05 | 0 | 0 |
| Refusal accuracy | ≥ 0.90 | 1.00 | 1.00 |
| False refusals (answerable) | — | 0.03 | 0.03 |
| Citation validity | 1.0 | 1.0 | 1.0 |
| Median response time | < 4 s | 0.82 s | 0.76 s |
| Eval cost | — | $0.011 | $0.011 |

The only recall difference is one multi-page question ("Which operators support 2B and 9S?"), so the default
stays `text-embedding-3-small`. The one false refusal is the model saying the wiki doesn't connect Pascal's
village and the Resistance. The run also found spoiler-tag errors the keyword checks missed (Tower and Route C
chapter references tagged too low); they're fixed in `data/spoiler_overrides.yaml` and now caught by
`scripts/audit_spoilers.py`. Trap answers are worth reading by hand after tag changes: leak terms only catch
the leaks you thought of.

## Attribution

Lore content comes from the [NieR Wiki](https://nier.fandom.com) on Fandom and is licensed under [CC BY-SA](https://creativecommons.org/licenses/by-sa/3.0/). Every answer cites and links its source sections.
