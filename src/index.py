"""Chroma collection per embedder (SPEC §8.2) and chunk upserts."""

from __future__ import annotations

import re
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from src.chunking import Chunk
from src.providers.base import Embedder


class IndexMismatchError(RuntimeError):
    pass


def collection_name(embedder_name: str) -> str:
    # "openai/text-embedding-3-small" -> "chunks__openai_text-embedding-3-small"
    return "chunks__" + re.sub(r"[^A-Za-z0-9._-]", "_", embedder_name)


def open_client(path: str | Path) -> ClientAPI:
    return chromadb.PersistentClient(path=str(path))


def get_collection(client: ClientAPI, embedder: Embedder, *, create: bool = False) -> Collection:
    """Return this embedder's collection, failing loudly if it was built with a different model."""
    name = collection_name(embedder.name)
    if create:
        col = client.get_or_create_collection(
            name,
            metadata={"embedder": embedder.name, "dim": embedder.dim},
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )
    else:
        try:
            col = client.get_collection(name, embedding_function=None)
        except Exception as e:
            raise IndexMismatchError(
                f"No index for {embedder.name}; run `scripts/ingest.py --reindex`."
            ) from e

    meta = col.metadata or {}
    if meta.get("embedder") != embedder.name or meta.get("dim") != embedder.dim:
        raise IndexMismatchError(
            f"Index was built with {meta.get('embedder')} (dim {meta.get('dim')}), "
            f"but config uses {embedder.name} (dim {embedder.dim}); "
            "run `scripts/ingest.py --reindex`."
        )
    return col


def rebuild_collection(client: ClientAPI, embedder: Embedder) -> Collection:
    """Drop and recreate this embedder's collection, so chunks removed from the corpus don't linger."""
    name = collection_name(embedder.name)
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    return get_collection(client, embedder, create=True)


def upsert_chunks(col: Collection, embedder: Embedder, chunks: list[Chunk], batch: int = 256) -> None:
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        col.upsert(
            ids=[c.chunk_id for c in part],
            documents=[c.text for c in part],
            embeddings=embedder.embed([c.text for c in part]),
            metadatas=[c.chroma_metadata() for c in part],
        )
