"""Page selection (SPEC §5.2) -> data/manifest.json."""

from __future__ import annotations

import re
from typing import Any

import mwparserfromhell as mw

from src.ingest.dump import Wiki, normalize_title


def descendants(wiki: Wiki, roots: set[str]) -> set[str]:
    children: dict[str, set[str]] = {}
    for cat, parents in wiki.category_parents.items():
        for p in parents:
            children.setdefault(p, set()).add(cat)
    out, stack = set(), list(roots)
    while stack:
        c = stack.pop()
        if c not in out:
            out.add(c)
            stack.extend(children.get(c, ()))
    return out


GAME_FIELDS = re.compile(r"(first_appearance|other_appearance\d*|appears|appearances?|game|games)")


def infobox_appearances(text: str) -> set[str]:
    """Games named in an infobox's game fields (first_appearance, appears, game, ...)."""
    games = set()
    for t in mw.parse(text).filter_templates(recursive=False):
        if "infobox" not in str(t.name).lower():
            continue
        for p in t.params:
            if GAME_FIELDS.fullmatch(str(p.name).strip()):
                games.update(g.strip() for g in re.split(r"[\n,;]|<br\s*/?>", mw.parse(p.value).strip_code()) if g.strip())
    return games


def select_pages(wiki: Wiki, sel: dict[str, Any]) -> list[dict[str, Any]]:
    root_re = re.compile(sel["category_root_pattern"])
    roots = {c for c in wiki.category_parents if root_re.search(c)}
    for p in wiki.articles.values():  # categories used on pages but lacking a category page
        roots |= {c for c in p.categories if root_re.search(c)}
    excluded = descendants(wiki, {normalize_title(c) for c in sel["exclude_categories"]})
    included = descendants(wiki, roots) - excluded
    exclude_titles = {normalize_title(t) for t in sel["exclude_titles"]}
    include_titles = {normalize_title(t) for t in sel.get("include_titles", [])}
    game = sel["infobox_game"]

    main = wiki.articles[normalize_title(sel["main_page"])]
    linked = {wiki.resolve(str(l.title)) for l in mw.parse(main.text).filter_wikilinks()}
    linked.discard(None)

    reasons: dict[str, list[str]] = {}
    for title, page in wiki.articles.items():
        cats = set(page.categories)
        r = [f"category:{c}" for c in sorted(cats & included)]
        if title == main.title or title in linked:
            r.append("link:main_page")
        if game in infobox_appearances(page.text):
            r.append("infobox:appearance")
        if title in include_titles:
            r.append("manual")
        if not r or title in exclude_titles:
            continue
        # Drop pages that only qualify through out-of-scope categories.
        if cats & excluded and not cats & included:
            continue
        reasons[title] = r

    for title in list(reasons):  # character subpages like "X/Character Story"
        for sub in wiki.articles:
            if sub.startswith(title + "/") and sub not in reasons and sub.rsplit("/", 1)[1] not in sel["skip_subpages"]:
                reasons[sub] = [f"subpage:{title}"]

    return [
        {"page_id": wiki.articles[t].page_id, "title": t,
         "revision_id": wiki.articles[t].revision_id, "reasons": reasons[t]}
        for t in sorted(reasons)
    ]
