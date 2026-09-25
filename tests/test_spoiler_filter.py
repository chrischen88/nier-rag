"""Required invariant (SPEC §12.1): no retrieved chunk is above the user's progress level."""

import random
import uuid

import chromadb
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.chunking import Chunk, ContentType
from src.index import get_collection, upsert_chunks
from src.retrieve import query, spoiler_where
from src.spoilers import MAX_LEVEL
from tests.conftest import FakeEmbedder

CONTENT_TYPES = [c.value for c in ContentType]


@pytest.fixture(scope="module")
def collection():
    rng = random.Random(0)
    emb = FakeEmbedder(f"fake/{uuid.uuid4().hex[:8]}", 8)
    col = get_collection(chromadb.EphemeralClient(), emb, create=True)
    chunks = [
        Chunk(
            chunk_id=f"{i}:0:0", page_title=f"Page {i}", section_path=f"Page {i} > Story",
            url=f"https://nier.fandom.com/wiki/Page_{i}", text=f"chunk {i} {rng.random()}",
            content_type=rng.choice(CONTENT_TYPES), spoiler_level=rng.randint(0, MAX_LEVEL),
            spoiler_source="default", revision_id=1, is_speculation=rng.random() < 0.2,
        )
        for i in range(300)
    ]
    upsert_chunks(col, emb, chunks)
    return col


vectors = st.lists(st.floats(-1, 1, allow_nan=False), min_size=8, max_size=8).filter(
    lambda v: any(abs(x) > 1e-3 for x in v))


@settings(max_examples=200, deadline=None)
@given(vec=vectors, level=st.integers(0, MAX_LEVEL), n=st.integers(1, 40),
       types=st.none() | st.lists(st.sampled_from(CONTENT_TYPES), min_size=1, unique=True),
       hide_spec=st.booleans())
def test_no_chunk_above_user_level(collection, vec, level, n, types, hide_spec):
    results = query(collection, vec, level, n=n, content_types=types, hide_speculation=hide_spec)
    for r in results:
        assert r.metadata["spoiler_level"] <= level
        if types:
            assert r.metadata["content_type"] in types
        if hide_spec:
            assert r.metadata["is_speculation"] is False


def test_every_level_returns_something(collection):
    for level in range(MAX_LEVEL + 1):
        assert query(collection, [0.1] * 8, level, n=5)


def test_filter_always_includes_spoiler_clause():
    assert spoiler_where(2) == {"spoiler_level": {"$lte": 2}}
    combined = spoiler_where(2, content_types=["story"], hide_speculation=True)
    assert {"spoiler_level": {"$lte": 2}} in combined["$and"]
    with pytest.raises(ValueError):
        spoiler_where(6)
