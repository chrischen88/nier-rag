import hashlib

import pytest


class FakeEmbedder:
    """Deterministic embedder for tests; no network."""

    def __init__(self, name: str = "fake/hash-8", dim: int = 8):
        self.name, self.dim = name, dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append([b / 255 - 0.5 for b in h[: self.dim]])
        return out


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
