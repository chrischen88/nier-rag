"""Spoiler-filtered retrieval (SPEC §7.3, §8.3): top-k from Chroma, then MMR. The refusal
threshold is applied by the caller (M3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from chromadb.api.models.Collection import Collection

from src.providers.base import Embedder
from src.spoilers import MAX_LEVEL, MIN_LEVEL


@dataclass
class Retrieved:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    similarity: float
    embedding: list[float] | None = field(default=None, repr=False)


def spoiler_where(user_level: int, *, content_types: list[str] | None = None,
                  hide_speculation: bool = False) -> dict[str, Any]:
    """Chroma `where` filter. The spoiler clause is always present: this is the hard guarantee."""
    if not MIN_LEVEL <= user_level <= MAX_LEVEL:
        raise ValueError(f"user_level must be {MIN_LEVEL}..{MAX_LEVEL}, got {user_level}")
    clauses: list[dict[str, Any]] = [{"spoiler_level": {"$lte": user_level}}]
    if content_types:
        clauses.append({"content_type": {"$in": list(content_types)}})
    if hide_speculation:
        clauses.append({"is_speculation": False})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def query(col: Collection, query_embedding: list[float], user_level: int, *, n: int = 20,
          content_types: list[str] | None = None, hide_speculation: bool = False) -> list[Retrieved]:
    res = col.query(
        query_embeddings=[query_embedding],
        n_results=n,
        where=spoiler_where(user_level, content_types=content_types, hide_speculation=hide_speculation),
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    return [
        Retrieved(chunk_id=i, text=d, metadata=m, similarity=1.0 - dist, embedding=list(e))  # cosine
        for i, d, m, dist, e in zip(res["ids"][0], res["documents"][0], res["metadatas"][0],
                                    res["distances"][0], res["embeddings"][0])
    ]


def mmr(query_embedding: list[float], candidates: list[Retrieved], k: int, lam: float = 0.5) -> list[Retrieved]:
    """Maximal marginal relevance: trade query similarity against similarity to already-picked chunks."""
    if len(candidates) <= k:
        return candidates
    vecs = np.array([c.embedding for c in candidates], dtype=float)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    q = np.asarray(query_embedding, dtype=float)
    rel = vecs @ (q / np.linalg.norm(q))
    picked: list[int] = []
    rest = list(range(len(candidates)))
    while len(picked) < k:
        if picked:
            redundancy = (vecs[rest] @ vecs[picked].T).max(axis=1)
        else:
            redundancy = np.zeros(len(rest))
        scores = lam * rel[rest] - (1 - lam) * redundancy
        picked.append(rest.pop(int(scores.argmax())))
    return [candidates[i] for i in picked]


def retrieve(col: Collection, embedder: Embedder, question: str, user_level: int,
             retrieval_cfg: dict[str, Any], **filters) -> list[Retrieved]:
    q = embedder.embed([question])[0]
    candidates = query(col, q, user_level, n=retrieval_cfg["candidates"], **filters)
    return mmr(q, candidates, retrieval_cfg["final_k"], retrieval_cfg["mmr_lambda"])
