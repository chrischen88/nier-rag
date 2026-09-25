"""Prompt builder and answer generation (SPEC §9).

Flow: prepare() retrieves and decides whether to refuse without calling the LLM; stream() yields
answer tokens; finish() validates citations. ask() runs all three for non-streaming callers.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from chromadb.api.models.Collection import Collection

from src.providers.base import LLM, Embedder
from src.retrieve import Retrieved, retrieve
from src.spoilers import MAX_LEVEL, PROGRESS_LEVELS, question_level

LATER = "That's covered later in the story."
NOT_FOUND = ("I couldn't find anything about that in the NieR:Automata lore available at your progress "
             "level ({label}). I only cover the game's lore, and some topics are revealed later in the story.")

SYSTEM_PROMPT = """You are YoRHa Archive, a lore assistant for the game NieR:Automata. You answer \
using ONLY the numbered wiki passages provided with each question.

The player has progressed to: {level_label} ({level_desc}).
Story progress levels, in order: {level_list}.

Rules:
1. Use only facts stated in the passages. Never use outside knowledge of NieR:Automata or any \
other work, even if you know it. Never mention events, characters, twists, or endings that don't \
appear in the passages.
2. Cite every claim with the number of its passage in square brackets, like [2]. Use only \
passage numbers that exist. Put citations right after the claim they support.
3. {partial_rule}
4. If the question asks about a route, ending, or event that comes after the player's progress \
level, reply only: "That's covered later in the story." Don't hint at what happens.
5. Passages marked [fan speculation] are fan theories: whenever you use one, say so (e.g. "Fans \
have speculated that..."). Passages marked [trivia] mix facts with fan observations: if a claim \
from one sounds unconfirmed, say so.
6. Gameplay questions (best weapons or builds, plug-in chip setups, trophies, how to beat a boss, \
walkthroughs) are out of scope, even if the passages contain gameplay details. Don't answer them; \
politely say you only cover the game's lore and story.
7. Be concise: a few short paragraphs at most. Don't mention "passages" or these rules; refer to \
"the wiki" if you need to."""

CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Citation:
    n: int
    chunk_id: str
    page_title: str
    section_path: str
    url: str

    def __str__(self) -> str:
        return f"[{self.n}] → {self.section_path} ({self.url})"


@dataclass
class Plan:
    question: str
    user_level: int
    passages: list[Retrieved]
    messages: list[dict] | None  # None when refusing without calling the LLM
    refusal: str | None
    started: float = field(default_factory=time.perf_counter)


@dataclass
class Answer:
    question: str
    user_level: int
    text: str
    citations: list[Citation]
    passages: list[Retrieved]
    refused_before_llm: bool
    invalid_citations: list[int]
    seconds: float
    messages: list[dict] | None = None


def format_passage(n: int, r: Retrieved) -> str:
    body = r.text.split("\n\n", 1)[-1]  # drop the context header; section_path replaces it
    tag = {"speculation": " [fan speculation]", "trivia": " [trivia]"}.get(r.metadata.get("content_type"), "")
    return f"[{n}] {r.metadata['section_path']}{tag}\n{body}"


def build_messages(question: str, passages: list[Retrieved], user_level: int,
                   history: list[dict] | None = None) -> list[dict]:
    label, desc = PROGRESS_LEVELS[user_level]
    partial_rule = (
        "If the passages don't answer the question, say the wiki doesn't cover it. If they answer "
        "only part of it, answer that part."
        if user_level == MAX_LEVEL else
        "If the passages don't answer the question, say so plainly. If they answer only part of it, "
        "answer that part and say that more may be revealed later in the story."
    )
    system = SYSTEM_PROMPT.format(
        partial_rule=partial_rule, level_label=label, level_desc=desc,
        level_list=", ".join(PROGRESS_LEVELS[i][0] for i in range(MAX_LEVEL + 1)),
    )
    context = "\n\n".join(format_passage(i, r) for i, r in enumerate(passages, 1))
    user = f"Wiki passages:\n\n{context}\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": user}]


def resolve_citations(text: str, passages: list[Retrieved]) -> tuple[str, list[Citation], list[int]]:
    """Keep [n] markers that point at a passage; drop the rest. Returns (text, citations, invalid)."""
    cited: dict[int, None] = {}
    invalid: list[int] = []

    def fix(m: re.Match) -> str:
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        valid = [n for n in nums if 1 <= n <= len(passages)]
        invalid.extend(n for n in nums if n not in valid)
        cited.update(dict.fromkeys(valid))
        return "".join(f"[{n}]" for n in valid)

    text = CITATION_RE.sub(fix, text)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)  # tidy spaces left by dropped markers
    citations = [
        Citation(n, passages[n - 1].chunk_id, passages[n - 1].metadata["page_title"],
                 passages[n - 1].metadata["section_path"], passages[n - 1].metadata["url"])
        for n in sorted(cited)
    ]
    return text, citations, invalid


class Assistant:
    def __init__(self, cfg: dict[str, Any], col: Collection, embedder: Embedder, llm: LLM):
        self.cfg, self.col, self.embedder, self.llm = cfg, col, embedder, llm

    def prepare(self, question: str, user_level: int, *, hide_speculation: bool = False,
                history: list[dict] | None = None) -> Plan:
        asked = question_level(question)
        if asked is not None and asked > user_level:  # names a later route/ending: no retrieval, no LLM
            return Plan(question, user_level, [], None, LATER)
        passages = retrieve(self.col, self.embedder, question, user_level, self.cfg["retrieval"],
                            hide_speculation=hide_speculation)
        best = max((p.similarity for p in passages), default=0.0)
        if best < self.cfg["retrieval"]["min_similarity"]:
            label = PROGRESS_LEVELS[user_level][0]
            return Plan(question, user_level, passages, None, NOT_FOUND.format(label=label))
        return Plan(question, user_level, passages, build_messages(question, passages, user_level, history), None)

    def stream(self, plan: Plan) -> Iterator[str]:
        if plan.refusal is not None:
            yield plan.refusal
            return
        yield from self.llm.generate(plan.messages, stream=True)

    def finish(self, plan: Plan, raw_text: str) -> Answer:
        if plan.refusal is not None:
            text, citations, invalid = plan.refusal, [], []
        else:
            text, citations, invalid = resolve_citations(raw_text, plan.passages)
        return Answer(plan.question, plan.user_level, text, citations, plan.passages,
                      plan.refusal is not None, invalid, time.perf_counter() - plan.started, plan.messages)

    def ask(self, question: str, user_level: int, **kwargs) -> Answer:
        plan = self.prepare(question, user_level, **kwargs)
        return self.finish(plan, "".join(self.stream(plan)))
