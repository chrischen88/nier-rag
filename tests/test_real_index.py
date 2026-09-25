"""Spoiler-filter invariant on the real index built by scripts/ingest.py. Uses stored chunk
embeddings as query vectors, so it needs no API key. Skipped if the index hasn't been built."""

import random

import pytest

from src.config import load_config, resolve_path
from src.index import collection_name
from src.retrieve import query
from src.spoilers import MAX_LEVEL


@pytest.fixture(scope="module")
def collection():
    cfg = load_config()
    path = resolve_path(cfg, "chroma")
    if not path.exists():
        pytest.skip("no index; run scripts/ingest.py")
    import chromadb
    client = chromadb.PersistentClient(path=str(path))
    name = collection_name(f"openai/{cfg['embeddings']['model']}")
    if name not in [c.name for c in client.list_collections()]:
        pytest.skip(f"no collection {name}")
    return client.get_collection(name, embedding_function=None)


def test_real_index_never_returns_chunks_above_user_level(collection):
    sample = collection.get(include=["embeddings"], limit=collection.count())
    rng = random.Random(0)
    vectors = rng.sample(list(sample["embeddings"]), 60)
    for level in range(MAX_LEVEL + 1):
        for v in vectors:
            for r in query(collection, list(v), level, n=20):
                assert r.metadata["spoiler_level"] <= level, (r.chunk_id, r.metadata["spoiler_level"], level)


def test_every_level_has_content(collection):
    for level in range(MAX_LEVEL + 1):
        assert collection.get(where={"spoiler_level": level}, limit=1)["ids"], f"no chunks at level {level}"
