import uuid

import chromadb
import pytest

from src.index import IndexMismatchError, collection_name, get_collection
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
