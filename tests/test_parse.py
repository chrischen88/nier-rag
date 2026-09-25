from src.ingest.dump import RawPage, Wiki
from src.ingest.parse import Renderer, parse_page, preprocess, split_sections

WIKI = Wiki(
    articles={t: None for t in ["YoRHa", "YoRHa No.9 Type S", "Pascal", "Test Page"]},
    redirects={"9S": "YoRHa No.9 Type S"},
    category_parents={},
)

PAGE = """{{Character Tabs}}
{{Character Infobox
<!-- Main -->
|image = <gallery>
X.png|Automata
</gallery>
|type = Protagonist
|race_note = [[YoRHa]]
|va_EN = {{w|Kira Buckland}}
}}
{{Q|Everything that lives is designed to end.|2B in the Prologue}}
'''{{PAGENAME}}''' is a [[YoRHa]] android.<ref>World Guide</ref> She works with [[9S|a Scanner]].
[[File:2B.png|thumb|A caption that should vanish]]
==Background==
===NieR:Automata===
Story text.{{Note|a footnote}}
{{#tag:tabber|A=
Ending A text.
{{!}}-{{!}}E=
Ending E text with [[Pascal]].
}}
==Gallery==
<gallery>
Y.png|Y
</gallery>
==Trivia==
* A fact.
[[Category:NieR:Automata Characters]]
[[ja:ヨルハ二号B型]]
"""


def parse(text=PAGE):
    page = RawPage(page_id=1, title="Test Page", ns=0, revision_id=7, timestamp="", text=text, redirect=None)
    return parse_page(page, WIKI, "https://nier.fandom.com", ["2B"], Renderer())


def test_sections_and_tabs():
    paths = [" > ".join(s.headings) for s in parse().sections]
    assert paths == ["", "Background > NieR:Automata", "Background > NieR:Automata > A",
                     "Background > NieR:Automata > E", "Trivia"]


def test_lead_is_clean():
    lead = parse().sections[0].text
    assert lead.startswith('"Everything that lives is designed to end." — 2B in the Prologue')
    assert "Test Page is a YoRHa android. She works with a Scanner." in lead
    for junk in ("World Guide", "caption", "Category", "ヨルハ", "Character Tabs"):
        assert junk not in lead


def test_links_resolve_through_redirects():
    p = parse()
    assert p.sections[0].links_to == ["YoRHa", "YoRHa No.9 Type S"]
    assert "Pascal" in p.links_to


def test_infobox_extracted_and_removed():
    p = parse()
    assert p.infobox_type == "character infobox"
    assert p.infobox == {"type": "Protagonist", "va_EN": "Kira Buckland"}
    assert all("Protagonist" not in s.text for s in p.sections)


def test_metadata():
    p = parse()
    assert p.url == "https://nier.fandom.com/wiki/Test_Page"
    assert p.categories == ["NieR:Automata Characters"]
    assert p.aliases == ["2B"]


def test_tabber_tag_form_nests_under_last_heading():
    rendered = Renderer().render(preprocess("==Endings==\n<tabber>A=first\n|-|\nB=second</tabber>"))
    secs = split_sections(rendered)[1:]
    assert [s.headings for s in secs] == [["Endings"], ["Endings", "A"], ["Endings", "B"]]
    assert [s.is_tab for s in secs] == [False, True, True]
    assert {s.anchor for s in secs} == {"Endings"}


def test_colon_links_and_displaytitle_and_spoiler_banners():
    p = parse("{{DISPLAYTITLE:[Top Secret] X}}\n{{Spoiler|NA|Route=B}}"
              "'''X''' is a friendly [[:Category:NieR:Automata Units|unit]] in [[YoRHa]].")
    assert p.sections[0].text == "X is a friendly unit in YoRHa."
    assert p.spoiler_banners == [{"game": "NA", "route": "B"}]


def test_tabber_inside_table_and_poem():
    p = parse("==Weapon Story==\n{|\n|<tabber>Level 1=\nDad, <poem>if you're cold</poem>\n|-|Level 2=\nMore.</tabber>\n|}")
    assert [s.headings for s in p.sections] == [["Weapon Story", "Level 1"], ["Weapon Story", "Level 2"]]
    assert all("<" not in s.text for s in p.sections)
