"""Streamlit UI (SPEC §10) driven through AppTest with fake providers and retrieval; no network."""

from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.config import ROOT
from src.index import IndexMismatchError
from src.providers import ProviderError
from src.providers.openai import Usage
from src.retrieve import Retrieved
from tests.conftest import FakeEmbedder


def passage(i, level=1, similarity=0.6):
    return Retrieved(f"1:{i}:0", f"Page {i} > Story\n\nBody of passage {i}.",
                     {"page_title": f"Page {i}", "section_path": f"Page {i} > Story", "url": f"https://x/{i}",
                      "spoiler_level": level, "content_type": "character", "is_speculation": False},
                     similarity, embedding=[0.1] * 8)


class FakeLLM:
    name = "fake/llm"

    def __init__(self, reply="Pascal leads a village [2]. Also [9].", fail=False):
        self.reply, self.fail, self.usage = reply, fail, Usage()

    def generate(self, messages, *, stream=False, **kw):
        def tokens():
            if self.fail:
                raise ProviderError("OpenAI request failed: rate limited")
            self.usage.prompt_tokens += 100
            self.usage.completion_tokens += 10
            yield from (self.reply[:6], self.reply[6:])
        return tokens()


@pytest.fixture
def app(monkeypatch):
    """Return a factory for an AppTest whose providers, index, and retrieval are fakes."""
    def make(llm=None, passages=None, index_error=None):
        embedder = FakeEmbedder()
        embedder.usage = Usage()

        def get_collection(client, emb):
            if index_error:
                raise index_error
            return SimpleNamespace(name="chunks__fake", count=lambda: 3)

        def retrieve(col, emb, question, level, cfg, **kw):
            emb.usage.embedding_tokens += 5
            return [p for p in (passages or [passage(1), passage(2)]) if p.metadata["spoiler_level"] <= level]

        monkeypatch.setattr("src.providers.get_embedder", lambda cfg: embedder)
        monkeypatch.setattr("src.providers.get_llm", lambda cfg: llm or FakeLLM())
        monkeypatch.setattr("src.index.open_client", lambda path: None)
        monkeypatch.setattr("src.index.get_collection", get_collection)
        monkeypatch.setattr("src.generate.retrieve", retrieve)
        st.cache_resource.clear()
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=10).run()
    yield make
    st.cache_resource.clear()


def ask(at, question, level=None):
    if level is not None:
        at.select_slider(key="progress_level").set_value(level).run()
    at.chat_input[0].set_value(question).run()
    assert not at.exception, at.exception
    return at


def test_defaults_to_the_most_spoiler_safe_level(app):
    at = app()
    assert not at.exception
    assert at.select_slider(key="progress_level").value == 0
    assert "chunks__fake" in at.sidebar.caption[1].value


def test_answer_shows_resolved_citations_with_expandable_passages(app):
    at = ask(app(), "Who is Pascal?", level=5)
    msgs = at.chat_message
    assert [m.name for m in msgs] == ["user", "assistant"]
    assert msgs[1].markdown[0].value == "Pascal leads a village [2]. Also."  # [9] doesn't exist
    [exp] = msgs[1].expander
    assert exp.label == "[2] Page 2 > Story"
    assert "Body of passage 2." in exp.markdown[0].value and "https://x/2" in exp.markdown[1].value


def test_session_cost_is_tracked(app):
    at = ask(app(), "Who is Pascal?", level=5)
    ask(at, "Who is Adam?")
    assert "220 chat tokens, 10 embedding tokens" in at.sidebar.caption[-1].value


def test_later_route_question_is_refused_without_the_llm(app):
    llm = FakeLLM()
    at = ask(app(llm=llm), "What happens in Ending E?", level=2)
    assert at.chat_message[1].markdown[0].value == "That's covered later in the story."
    assert llm.usage.prompt_tokens == 0


def test_lowering_progress_hides_turns_asked_at_a_later_level(app):
    at = ask(app(), "Who is Pascal?", level=4)
    at.select_slider(key="progress_level").set_value(1).run()
    assert not at.chat_message
    assert "1 earlier question hidden" in at.caption[0].value
    at.select_slider(key="progress_level").set_value(4).run()
    assert len(at.chat_message) == 2


def test_debug_panel_shows_passages_levels_and_prompt(app):
    at = app()
    at.toggle(key="debug").set_value(True).run()
    ask(at, "Who is Pascal?", level=3)
    debug = [e for e in at.chat_message[1].expander if e.label == "Debug"][0]
    table = debug.dataframe[0].value
    assert list(table["level"]) == [1, 1] and list(table["#"]) == [1, 2]
    assert "Removed citations" in debug.warning[0].value
    assert any("Question: Who is Pascal?" in c.value for c in debug.code)


def test_provider_failure_is_shown_not_raised(app):
    at = ask(app(llm=FakeLLM(fail=True)), "Who is Pascal?", level=5)
    assert "rate limited" in at.chat_message[1].error[0].value
    at.run()  # the error stays in the history on rerun
    assert "rate limited" in at.chat_message[1].error[0].value


def test_index_mismatch_is_reported_at_startup(app):
    at = app(index_error=IndexMismatchError("Index was built with X; run `scripts/ingest.py --reindex`."))
    assert not at.exception
    assert "--reindex" in at.error[0].value
    assert not at.chat_input
