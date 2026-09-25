import json

import pytest

from src.chunking import Chunk
from src.spoilers import (LLMClassifier, SpoilerTagger, banner_level, heading_marker_level, load_overrides,
                          write_report)

CFG = {"heading_rules": [{"pattern": "^Other [Aa]ppearances$", "level": 5},
                         {"pattern": "^Chapter 01\\b", "level": 0}],
       "category_rules": [{"pattern": "^Colosseums$", "level": 5}]}


def chunk(path="2B > Background", cid="1:1:0", categories=(), banners=(), text=None):
    return Chunk(chunk_id=cid, page_title=path.split(" > ")[0], section_path=path, url="https://x",
                 text=text or f"{path}\n\nSome text.", content_type="character", spoiler_level=5,
                 spoiler_source="default", revision_id=1, categories=list(categories),
                 spoiler_banners=list(banners))


class FakeLLM:
    name = "fake/llm"

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def generate(self, messages, **kw):
        self.calls += 1
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


def tagger(tmp_path, reply=None, overrides=None, threshold=0.7):
    clf = LLMClassifier(FakeLLM(reply), threshold, tmp_path / "cache.jsonl") if reply is not None else None
    return SpoilerTagger(CFG, overrides or {}, clf)


@pytest.mark.parametrize("headings,level", [
    (["2B", "Background", "NieR:Automata", "Endings", "C"], 3),
    (["A2", "Background", "Endings", "E"], 4),
    (["Adam", "Gameplay", "Route B"], 2),
    (["X", "Routes C and D"], 3),
    (["X", "Ending A"], 1),
    (["Pascal", "Background", "Village Route"], None),
    (["NieR:Automata/Endings", "K"], None),  # joke endings F-Z go to the classifier
    (["X", "Story", "C"], None),  # a bare letter only counts under an Endings heading
    (["The (E)nd of YoRHa", "Gameplay"], 4),  # ending page titles carry their letter
    (["Or not to (B)e"], 2),
    (["(L)one wolf"], None),  # hidden endings F-Z go to the classifier
])
def test_heading_markers(headings, level):
    assert heading_marker_level(headings) == level


@pytest.mark.parametrize("banners,level", [
    ([{"game": "NA", "route": "C/D"}], 3),
    ([{"game": "NA", "route": "A,C/D"}], 3),
    ([{"game": "NieR: Automata", "route": "B"}], 2),
    ([{"game": "NA", "ending": "C,D"}], 3),
    ([{"game": "NA"}], None),  # no route given
    ([{"game": "N", "route": "A,B"}], None),  # another game's banner
])
def test_banner_levels(banners, level):
    assert banner_level(banners) == level


def test_rule_order(tmp_path):
    t = tagger(tmp_path, overrides={"2B > Other Appearances": 3})
    # manual beats heading rule
    assert (t.tag(chunk("2B > Other Appearances")).source, t.tag(chunk("2B > Other Appearances")).level) == ("manual", 3)
    # heading beats category/banner
    r = t.tag(chunk("2B > Background > Endings > A", banners=[{"game": "NA", "route": "C/D"}]))
    assert (r.source, r.level) == ("heading_rule", 1)
    r = t.tag(chunk("Bunker > Story", banners=[{"game": "NA", "route": "C/D"}]))
    assert (r.source, r.level) == ("category_rule", 3)
    r = t.tag(chunk("Arena", categories=["Colosseums"]))
    assert (r.source, r.level) == ("category_rule", 5)
    r = t.tag(chunk("Chapter 01: Prologue > Dialogue"))
    assert (r.source, r.level) == ("heading_rule", 0)


def test_override_keys(tmp_path):
    t = tagger(tmp_path, overrides={"9:9:0": 2, "Pascal": 1})
    assert t.tag(chunk("Anything", cid="9:9:0")).level == 2
    assert t.tag(chunk("Pascal > Trivia")).level == 1  # page title


def test_llm_accepted_above_threshold_and_cached(tmp_path):
    t = tagger(tmp_path, reply={"level": 2, "confidence": 0.9, "reason": "Route B reveal"})
    r = t.tag(chunk())
    assert (r.source, r.level, r.reason) == ("llm", 2, "Route B reveal")
    t2 = tagger(tmp_path, reply={"level": 0, "confidence": 1.0})  # same cache file
    assert t2.tag(chunk()).level == 2 and t2.classifier.llm.calls == 0


def test_low_confidence_or_garbage_defaults_to_5(tmp_path):
    r = tagger(tmp_path, reply={"level": 1, "confidence": 0.5}).tag(chunk())
    assert (r.source, r.level, r.llm_level) == ("default", 5, 1)
    r = tagger(tmp_path / "x", reply="not json").tag(chunk())
    assert (r.source, r.level) == ("default", 5)
    r = tagger(tmp_path / "y", reply={"level": 9, "confidence": 1.0}).tag(chunk())
    assert (r.source, r.level) == ("default", 5)


def test_no_classifier_defaults_to_5(tmp_path):
    assert tagger(tmp_path).tag(chunk()).level == 5


def test_tag_all_updates_chunks_and_report(tmp_path):
    (tmp_path / "y").mkdir()
    t = tagger(tmp_path / "y", reply={"level": 1, "confidence": 0.8, "reason": "early"})
    chunks = [chunk(), chunk("2B > Other Appearances > SINoALICE", cid="1:2:0")]
    t.tag_all(chunks, workers=2)
    assert [(c.spoiler_level, c.spoiler_source.value) for c in chunks] == [(1, "llm"), (5, "heading_rule")]
    results = [t.tag(c) for c in chunks]
    assert write_report(tmp_path / "r.csv", chunks, results) == 1  # only llm/default rows


def test_overrides_validation(tmp_path):
    p = tmp_path / "o.yaml"
    p.write_text("Pascal: 7\n")
    with pytest.raises(ValueError):
        load_overrides(p)
    p.write_text("{}\n")
    assert load_overrides(p) == {}
