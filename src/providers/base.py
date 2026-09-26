"""Provider interfaces and factories (SPEC §8.1). Nothing outside this package imports a provider directly."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from src.config import ConfigError


class ProviderError(RuntimeError):
    """A provider request failed after its retries (rate limit, server error, bad request...)."""


@runtime_checkable
class Embedder(Protocol):
    name: str  # e.g. "openai/text-embedding-3-small"
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class LLM(Protocol):
    name: str  # e.g. "openai/gpt-4o-mini"

    def generate(
        self,
        messages: list[dict],
        *,
        stream: bool = False,
        temperature: float | None = 0.2,
        json_mode: bool = False,
    ) -> str | Iterator[str]: ...


def get_embedder(cfg: dict[str, Any]) -> Embedder:
    from src.providers.openai import OpenAIEmbedder

    emb = cfg["embeddings"]
    return OpenAIEmbedder(model=emb["model"], batch_size=emb.get("batch_size", 256),
                          max_retries=cfg["llm"].get("max_retries", 5))


def get_llm(cfg: dict[str, Any], section: str = "llm") -> LLM:
    """Build the chat LLM (section="llm") or the spoiler classifier (section="classifier_llm").

    Settings missing from `section` fall back to the `llm` section.
    """
    from src.providers.openai import OpenAILLM

    settings = {**cfg["llm"], **cfg.get(section, {})}
    allowed = cfg.get("allowed_chat_models")
    if allowed is not None and settings["model"] not in allowed:
        raise ConfigError(f"{section}.model '{settings['model']}' is not in allowed_chat_models {allowed}")
    return OpenAILLM(
        model=settings["model"],
        max_output_tokens=settings.get("max_output_tokens", 800),
        default_temperature=settings.get("temperature"),
        max_retries=settings.get("max_retries", 5),
    )
