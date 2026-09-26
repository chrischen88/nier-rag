import json

import pytest

from src.config import ROOT
from src.evaluate import Result, load_questions, page_matches, render_report, score


def result(**kw):
    base = dict(id="q1", question="Who?", user_level=1, expected_pages=["Pascal"], should_refuse=False,
                leak_terms=[], retrieved=[{"page_title": "Pascal", "section_path": "Pascal", "spoiler_level": 1,
                                           "similarity": 0.6}],
                answer="Pascal leads a village [1].", citations=1, seconds=1.0)
    return Result(**{**base, **kw})


def test_subpages_count_as_their_page():
    assert page_matches("Desert Zone/Border Area", ["Desert Zone"])
    assert not page_matches("Desert Zone Map", ["Desert Zone"])


def test_hit_is_only_scored_for_answerable_questions_with_expected_pages():
    assert result().hit is True
    assert result(retrieved=[]).hit is False
    assert result(expected_pages=[]).hit is None
    assert result(should_refuse=True).hit is None


def test_leak_terms_match_whole_words_case_insensitively():
    r = result(leak_terms=["ark", "2E"], answer="The Tower is dark. Also 2e.")
    assert r.leaked_terms == ["2E"]


def test_refusal_needs_refusal_wording_and_no_citations():
    assert result(answer="I only cover the game's lore.", citations=0).refused
    assert result(answer="x", refused_before_llm=True).refused
    assert not result(answer="The wiki doesn't cover his past, but he leads a village [1].").refused
    assert result(answer=None).refused is None


def test_retrieval_leak_is_any_passage_above_the_user_level():
    r = result(retrieved=[{"page_title": "Tower", "section_path": "Tower", "spoiler_level": 3, "similarity": 0.5}])
    assert r.retrieval_leaks and score([r])[1].value == 1.0 and score([r])[1].passed is False


def test_metrics_and_targets():
    results = [
        result(),
        result(id="q2", should_refuse=True, expected_pages=[], answer="I only cover the game's lore.", citations=0),
        result(id="q3", leak_terms=["extinct"], expected_pages=[], answer="Humanity is extinct [1]."),
        result(id="q4", invalid_citations=[9]),
    ]
    m = {x.name: x for x in score(results)}
    assert m["Recall@k"].value == 1.0 and m["Recall@k"].n == 2
    assert m["Answer leak rate"].value == 1.0 and m["Answer leak rate"].passed is False
    assert m["Refusal accuracy"].value == 1.0 and m["Refusal accuracy"].passed
    assert m["Citation validity"].value == 0.75 and m["Citation validity"].passed is False
    assert m["Median response time (s)"].passed
    report = render_report(results, list(m.values()), {"timestamp": "t", "Chat model": "fake"})
    assert "| q3 | L1 | Who? | extinct |" in report and "❌" in report


def test_retrieval_only_scores_just_retrieval():
    names = [m.name for m in score([result(answer=None)])]
    assert names == ["Recall@k", "Retrieval leak rate"]


def test_load_questions_rejects_bad_items(tmp_path):
    p = tmp_path / "q.jsonl"
    item = {"id": "a", "question": "?", "user_level": 1, "expected_pages": [], "should_refuse": False,
            "leak_terms": []}
    p.write_text(json.dumps(item) + "\n" + json.dumps(item) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(p)
    p.write_text(json.dumps({**item, "user_level": 9}) + "\n")
    with pytest.raises(ValueError, match="out of range"):
        load_questions(p)


def test_question_set_meets_spec():
    """SPEC §11: 40-60 items, >= 15 spoiler traps, refusal items, expected pages that exist."""
    qs = load_questions(ROOT / "eval/questions.jsonl")
    assert 40 <= len(qs) <= 60
    assert sum(bool(q.leak_terms) for q in qs) >= 15
    assert any(q.should_refuse for q in qs)
    assert sum(len(q.expected_pages) > 1 for q in qs) >= 5  # multi-page questions
    titles = {p["title"] for p in json.load(open(ROOT / "data/manifest.json"))["pages"]}
    missing = {e for q in qs for e in q.expected_pages} - titles
    assert not missing, missing
