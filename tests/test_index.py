import uuid

import chromadb
import pytest

from src.chunking import Chunk, SpoilerSource
from src.index import (IndexMismatchError, collection_name, get_collection, index_collections, update_metadata,
                       upsert_chunks)
from tests.conftest import FakeEmbedder


def test_collection_name():
    assert collection_name("openai/text-embedding-3-small") == "chunks__openai_text-embedding-3-small"


def test_mismatched_embedder_is_rejected():
    client = chromadb.EphemeralClient()
    name = f"fake/{uuid.uuid4().hex[:8]}"
    get_collection(client, FakeEmbedder(name, 8), create=True)
    get_collection(client, FakeEmbedder(name, 8))  # same model: fine
    with pytest.raises(IndexMismatchError, match="--reindex"):
        get_collection(client, FakeEmbedder(name, 16))


def test_missing_index_is_reported():
    with pytest.raises(IndexMismatchError, match="--reindex"):
        get_collection(chromadb.EphemeralClient(), FakeEmbedder(f"fake/{uuid.uuid4().hex}", 8))


def test_retag_updates_metadata_in_every_index_without_reembedding():
    client = chromadb.EphemeralClient()
    chunks = [Chunk(chunk_id=f"{i}:0:0", page_title="P", section_path=f"P > S{i}", url="https://x", text=f"t{i}",
                    content_type="story", spoiler_level=1, spoiler_source="llm", revision_id=1) for i in range(3)]
    names = []
    for dim in (8, 16):
        emb = FakeEmbedder(f"fake/{uuid.uuid4().hex[:8]}", dim)
        upsert_chunks(get_collection(client, emb, create=True), emb, chunks)
        names.append(collection_name(emb.name))
    cols = [c for c in index_collections(client) if c.name in names]  # ephemeral clients share state
    assert len(cols) == 2
    before = [c.get(include=["embeddings"])["embeddings"].tolist() for c in cols]

    chunks[0].spoiler_level, chunks[0].spoiler_source = 3, SpoilerSource.MANUAL
    for col in cols:
        update_metadata(col, chunks)
    for col, emb_before in zip(cols, before):
        got = col.get(ids=["0:0:0"], include=["metadatas", "embeddings"])
        assert got["metadatas"][0]["spoiler_level"] == 3 and got["metadatas"][0]["spoiler_source"] == "manual"
    assert [c.get(include=["embeddings"])["embeddings"].tolist() for c in cols] == before

    with pytest.raises(IndexMismatchError, match="--reindex"):
        update_metadata(cols[0], chunks[:2])  # index holds a chunk that's gone from chunks.jsonl
