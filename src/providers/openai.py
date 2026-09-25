"""OpenAI implementations of Embedder and LLM: batching, retries, usage tracking."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from openai import OpenAI

from src.config import require_openai_key

# Known output sizes; unknown models are probed once on first use.
EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_UNSET = object()


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    requests: int = 0


def _client(max_retries: int) -> OpenAI:
    # The SDK retries rate-limit (429) and server (5xx) errors with exponential backoff.
    return OpenAI(api_key=require_openai_key(), max_retries=max_retries)


class OpenAIEmbedder:
    def __init__(self, model: str, batch_size: int = 256, max_retries: int = 5):
        self.model = model
        self.name = f"openai/{model}"
        self.batch_size = min(batch_size, 256)
        self.usage = Usage()
        self._client = _client(max_retries)
        self._dim = EMBEDDING_DIMS.get(model)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed(["dimension probe"])[0])
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self._client.embeddings.create(model=self.model, input=texts[i : i + self.batch_size])
            vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
            self.usage.embedding_tokens += resp.usage.total_tokens
            self.usage.requests += 1
        return vectors


class OpenAILLM:
    def __init__(self, model: str, max_output_tokens: int = 800,
                 default_temperature: float | None = 0.2, max_retries: int = 5):
        self.model = model
        self.name = f"openai/{model}"
        self.max_output_tokens = max_output_tokens
        self.default_temperature = default_temperature
        self.usage = Usage()
        self._client = _client(max_retries)

    def generate(self, messages: list[dict], *, stream: bool = False,
                 temperature: float | None | object = _UNSET,
                 json_mode: bool = False) -> str | Iterator[str]:
        temp = self.default_temperature if temperature is _UNSET else temperature
        kwargs: dict = {"model": self.model, "messages": messages,
                        "max_completion_tokens": self.max_output_tokens}
        if temp is not None:
            kwargs["temperature"] = temp
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if stream:
            return self._stream(kwargs)

        resp = self._client.chat.completions.create(**kwargs)
        self._record(resp.usage)
        return resp.choices[0].message.content or ""

    def _stream(self, kwargs: dict) -> Iterator[str]:
        chunks = self._client.chat.completions.create(
            **kwargs, stream=True, stream_options={"include_usage": True})
        for chunk in chunks:
            if chunk.usage:
                self._record(chunk.usage)
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _record(self, usage) -> None:
        self.usage.requests += 1
        if usage:
            self.usage.prompt_tokens += usage.prompt_tokens
            self.usage.completion_tokens += usage.completion_tokens


def estimate_cost(cfg: dict, llm_usage: Usage, embed_usage: Usage | None = None) -> float:
    """USD estimate from `pricing` in config.yaml (per 1M tokens)."""
    price = cfg["pricing"]
    cost = (llm_usage.prompt_tokens * price["chat_input"] + llm_usage.completion_tokens * price["chat_output"]) / 1e6
    if embed_usage is not None:
        cost += embed_usage.embedding_tokens * price["embedding"] / 1e6
    return cost
