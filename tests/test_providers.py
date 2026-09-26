import pytest

from src.config import ConfigError, load_config
from src.providers import Embedder, LLM, get_embedder, get_llm


def test_missing_key_fails_loudly(monkeypatch):
    cfg = load_config()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        get_llm(cfg)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        get_embedder(cfg)


def test_factories_build_from_config(openai_key):
    cfg = load_config()
    emb = get_embedder(cfg)
    assert isinstance(emb, Embedder)
    assert emb.name == f"openai/{cfg['embeddings']['model']}"
    assert emb.dim == 1536

    llm = get_llm(cfg)
    assert isinstance(llm, LLM)
    assert llm.name == f"openai/{cfg['llm']['model']}"


def test_only_allowed_chat_models(openai_key):
    cfg = load_config()
    assert cfg["llm"]["model"] == cfg["classifier_llm"]["model"] == "gpt-4o-mini"
    cfg["llm"]["model"] = "gpt-4o"
    with pytest.raises(ConfigError, match="allowed_chat_models"):
        get_llm(cfg)


def test_classifier_section_overrides_and_inherits(openai_key):
    cfg = load_config()
    cfg["allowed_chat_models"].append("some-other-model")
    cfg["classifier_llm"] = {"model": "some-other-model", "temperature": 0.0}
    clf = get_llm(cfg, section="classifier_llm")
    assert clf.name == "openai/some-other-model"
    assert clf.default_temperature == 0.0
    assert clf.max_output_tokens == cfg["llm"]["max_output_tokens"]  # inherited


@pytest.mark.parametrize("stream", [False, True])
def test_openai_errors_become_provider_errors(openai_key, stream):
    import httpx
    from openai import APIConnectionError

    from src.providers import ProviderError

    def fail(**kw):
        raise APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))

    llm = get_llm(load_config())
    llm._client.chat.completions.create = fail
    with pytest.raises(ProviderError, match="OpenAI request failed"):
        out = llm.generate([{"role": "user", "content": "hi"}], stream=stream)
        list(out) if stream else out
