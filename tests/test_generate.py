import pytest

from src.generate import GAMEPLAY, LATER, Assistant, build_messages, resolve_citations
from src.retrieve import Retrieved
from src.spoilers import question_level


def passage(i, similarity=0.6, content_type="character"):
    return Retrieved(f"1:{i}:0", f"Page {i} > Story\n\nText {i}.",
                     {"page_title": f"Page {i}", "section_path": f"Page {i} > Story", "url": f"https://x/{i}",
                      "spoiler_level": 1, "content_type": content_type,
                      "is_speculation": content_type in ("trivia", "speculation")}, similarity)


class FakeLLM:
    name = "fake/llm"

    def __init__(self, reply="Answer [1]."):
        self.reply, self.calls = reply, 0

    def generate(self, messages, *, stream=False, **kw):
        self.calls += 1
        return iter([self.reply[:5], self.reply[5:]]) if stream else self.reply


@pytest.fixture
def assistant(monkeypatch):
    def make(passages, reply="Answer [1]."):
        monkeypatch.setattr("src.generate.retrieve", lambda *a, **k: passages)
        cfg = {"retrieval": {"min_similarity": 0.35}}
        return Assistant(cfg, col=None, embedder=None, llm=FakeLLM(reply))
    return make


def test_messages_number_passages_and_mark_speculation():
    msgs = build_messages("Who?", [passage(1), passage(2, content_type="speculation")], user_level=1)
    assert msgs[0]["role"] == "system" and "Route A" in msgs[0]["content"]
    user = msgs[-1]["content"]
    assert "[1] Page 1 > Story\nText 1." in user
    assert "[2] Page 2 > Story [fan speculation]" in user
    assert user.endswith("Question: Who?")


def test_level_5_prompt_does_not_hint_at_later_content():
    assert "revealed later" not in build_messages("Q", [passage(1)], user_level=5)[0]["content"]
    assert "revealed later" in build_messages("Q", [passage(1)], user_level=1)[0]["content"]


def test_invalid_citations_are_removed():
    text, cites, invalid = resolve_citations("A [1]. B [2, 7]. C [9] .", [passage(1), passage(2)])
    assert text == "A [1]. B [2]. C."
    assert [c.n for c in cites] == [1, 2] and invalid == [7, 9]
    assert cites[0].url == "https://x/1"


def test_low_similarity_refuses_without_llm(assistant):
    a = assistant([passage(1, similarity=0.2)])
    ans = a.ask("How do I bake bread?", 5)
    assert ans.refused_before_llm and a.llm.calls == 0 and not ans.citations


def test_later_route_question_refuses_without_retrieval_or_llm(assistant):
    a = assistant([passage(1)])
    ans = a.ask("What happens in Ending E?", 2)
    assert ans.text == LATER and ans.refused_before_llm and a.llm.calls == 0 and ans.passages == []


def test_streamed_answer_matches_citations(assistant):
    a = assistant([passage(1), passage(2)], reply="Pascal leads a village [2].")
    plan = a.prepare("Who is Pascal?", 5)
    ans = a.finish(plan, "".join(a.stream(plan)))
    assert ans.text == "Pascal leads a village [2]." and [c.n for c in ans.citations] == [2]


@pytest.mark.parametrize("q,level", [
    ("What happens in Ending E?", 4), ("Explain route C/D", 3), ("Is Ending D sad?", 3),
    ("What changes in Route B?", 2), ("ending a", 1), ("Who is Pascal?", None),
    ("What's the weapon ending of the story?", None),
])
def test_question_level(q, level):
    assert question_level(q) == level


@pytest.mark.parametrize("q", ["Where can I farm Titanium Alloy?", "What's the best plug-in chip setup?",
                               "How do I beat Engels?", "How do I get all the trophies?"])
def test_gameplay_questions_refuse_without_retrieval_or_llm(assistant, q):
    a = assistant([passage(1)])
    ans = a.ask(q, 5)
    assert ans.text == GAMEPLAY and ans.refused_before_llm and a.llm.calls == 0 and ans.passages == []


@pytest.mark.parametrize("q", ["Who is 9S's best friend?", "What weapons does 2B use?", "Where is Pascal's village?",
                               "Who built the Tower?"])
def test_lore_questions_are_not_gameplay(assistant, q):
    a = assistant([passage(1)])
    assert not a.ask(q, 5).refused_before_llm
