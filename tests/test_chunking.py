import pytest

from src.chunking import (Chunk, ContentType, chunk_page, count_tokens, merge_short_sections,
                          split_text, validate_chunks)

CFG = {"max_tokens": 60, "overlap_tokens": 15, "min_tokens": 10,
       "display_names": {"YoRHa No.2 Type B": "2B"}}


def sec(headings, text, is_tab=False, anchor=None):
    return {"headings": headings, "text": text, "links_to": [], "is_tab": is_tab,
            "anchor": anchor or (headings[-1] if headings else None)}


def page(sections, **kw):
    return {"page_id": 42, "title": "YoRHa No.2 Type B", "revision_id": 9,
            "url": "https://nier.fandom.com/wiki/YoRHa_No.2_Type_B", "categories": ["Androids"],
            "infobox_type": "character infobox", "infobox": {"type": "Protagonist"},
            "sections": sections, **kw}


def test_short_sections_merge_into_parent():
    out = merge_short_sections([sec([], "Lead text that is long enough to stand alone here."),
                                sec(["Description"], "Tiny.")], min_tokens=10)
    assert len(out) == 1 and out[0]["text"].endswith("Description: Tiny.")


def test_short_siblings_merge_when_parent_has_no_text():
    out = merge_short_sections([sec(["Background"], "Long enough background text to keep on its own."),
                                sec(["Background", "One"], "Tiny."), sec(["Background", "Two"], "Also tiny.")],
                               min_tokens=10)
    assert [s["headings"] for s in out] == [["Background"]]


def test_route_tabs_and_trivia_never_merge():
    secs = [sec(["Story"], "Story text that is definitely long enough to stand alone."),
            sec(["Story", "B"], "Short.", is_tab=True),
            sec(["Story", "Route C"], "Short."),
            sec(["Trivia"], "Short.")]
    assert len(merge_short_sections(secs, min_tokens=10)) == 4


def test_split_respects_limit_and_overlaps():
    text = "\n".join(f"Sentence number {i} is about the machines. Another clause {i} follows." for i in range(12))
    parts = split_text(text, max_tokens=60, overlap_tokens=15)
    assert len(parts) > 1
    assert all(count_tokens(p) <= 60 for p in parts)
    for a, b in zip(parts, parts[1:]):
        assert b.split("\n")[0] in a  # next part starts with the tail of the previous one


def test_chunk_page_fields():
    chunks = chunk_page(page([sec([], "2B is an android. " * 3), sec(["Background", "NieR:Automata"], "Story. " * 80),
                              sec(["Trivia"], "A fact that is long enough to count as its own section.")]), CFG)
    validate_chunks(chunks)
    first = chunks[0]
    assert first.chunk_id == "42:0:0" and first.section_path == "2B"
    assert first.infobox == {"type": "Protagonist"} and all(c.infobox is None for c in chunks[1:])
    story = [c for c in chunks if c.section_path == "2B > Background > NieR:Automata"]
    assert len(story) > 1 and [c.chunk_id for c in story][:2] == ["42:1:0", "42:1:1"]
    assert story[0].url.endswith("#NieR:Automata")
    assert story[0].text.startswith("2B > Background > NieR:Automata\n\n")
    trivia = chunks[-1]
    assert trivia.content_type == ContentType.TRIVIA and trivia.is_speculation
    assert all(c.spoiler_level == 5 and c.spoiler_source.value == "default" for c in chunks)


def test_validate_rejects_duplicates():
    c = chunk_page(page([sec([], "Some lead text for the page.")]), CFG)[0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_chunks([c, c])


def test_chunk_roundtrips_through_json():
    c = chunk_page(page([sec([], "Some lead text for the page.")]), CFG)[0]
    assert Chunk(**c.to_json()) == c


def test_infobox_allow_list_goes_into_lead():
    cfg = {**CFG, "infobox_text_fields": {"va_EN": "English voice"}}
    pg = page([sec([], "2B is an android.")], infobox={"va_EN": "Kira Buckland", "aka": "2E"})
    lead = chunk_page(pg, cfg)[0].text
    assert "English voice: Kira Buckland" in lead and "2E" not in lead
