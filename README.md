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
uv run python scripts/ingest.py     # dump -> manifest -> parse -> chunk -> embed into Chroma
uv run python scripts/ingest.py --reindex    # re-embed data/chunks.jsonl only (e.g. after changing the embedder)
uv run python scripts/query.py "Who is Pascal?" --level 1   # inspect retrieval at a progress level
uv run python scripts/ask.py "Who is Pascal?" --level 1     # streamed answer with citations (--debug shows passages)
uv run streamlit run app.py         # M5
uv run python scripts/eval.py       # M6
```

## Attribution

Lore content comes from the [NieR Wiki](https://nier.fandom.com) on Fandom and is licensed under [CC BY-SA](https://creativecommons.org/licenses/by-sa/3.0/). Every answer cites and links its source sections.
