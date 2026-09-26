.PHONY: help install serve test ingest reindex retag audit eval

PORT ?= 8501

help:  ## list targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:  ## install deps
	uv sync

serve:  ## run the Streamlit app (PORT=8501)
	uv run streamlit run app.py --server.port $(PORT)

test:  ## run the test suite
	uv run pytest

ingest:  ## full ingest pipeline (calls OpenAI)
	uv run python scripts/ingest.py

reindex:  ## re-tag + re-embed data/chunks.jsonl (calls OpenAI)
	uv run python scripts/ingest.py --reindex

retag:  ## re-tag after editing overrides; updates every index, no re-embedding
	uv run python scripts/ingest.py --retag

audit:  ## flag chunks that state a twist below its reveal level
	uv run python scripts/audit_spoilers.py

eval:  ## full eval -> reports/eval_<timestamp>.md (calls OpenAI, ~$0.01)
	uv run python scripts/eval.py
