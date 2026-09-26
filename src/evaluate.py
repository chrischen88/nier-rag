"""Eval scoring (SPEC §11): per-question results, the metrics and their targets, and the report.

scripts/eval.py runs the questions; everything here is pure so it can be unit-tested.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.spoilers import MAX_LEVEL, MIN_LEVEL

REQUIRED_FIELDS = {"id", "question", "user_level", "expected_pages", "should_refuse", "leak_terms"}

# Ways the assistant declines: the two canned refusals (src/generate.py) and the phrasings the system
# prompt asks for. Only counted as a refusal when the answer also cites nothing.
REFUSAL_RE = re.compile(
    r"covered later in the story|couldn't find|could not find|only cover|(?:doesn't|does not|don't|do not) "
    r"cover|isn't covered|not covered|out of scope|no information|(?:doesn't|does not|don't) (?:say|mention)"
    r"|can't help|unable to (?:answer|help)|(?:doesn't|does not) (?:provide|contain|include)",
    re.I,
)


@dataclass
class Question:
    id: str
    question: str
    user_level: int
    expected_pages: list[str]
    should_refuse: bool
    leak_terms: list[str]
    reference: str = ""


def load_questions(path: Path) -> list[Question]:
    questions, seen = [], set()
    for n, line in enumerate(open(path), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        missing = REQUIRED_FIELDS - raw.keys()
        if missing:
            raise ValueError(f"{path}:{n}: missing {sorted(missing)}")
        q = Question(**{k: raw[k] for k in (*REQUIRED_FIELDS, "reference") if k in raw})
        if q.id in seen:
            raise ValueError(f"{path}:{n}: duplicate id {q.id}")
        if not MIN_LEVEL <= q.user_level <= MAX_LEVEL:
            raise ValueError(f"{path}:{n}: user_level {q.user_level} out of range")
        seen.add(q.id)
        questions.append(q)
    return questions


def page_matches(title: str, expected: list[str]) -> bool:
    """An expected page or one of its subpages ("Desert Zone/Border Area" counts for "Desert Zone")."""
    return any(title == e or title.startswith(e + "/") for e in expected)


@dataclass
class Result:
    id: str
    question: str
    user_level: int
    expected_pages: list[str]
    should_refuse: bool
    leak_terms: list[str]
    retrieved: list[dict[str, Any]]  # page_title, section_path, spoiler_level, similarity
    answer: str | None = None  # None in --retrieval-only runs
    refused_before_llm: bool = False
    citations: int = 0
    invalid_citations: list[int] = field(default_factory=list)
    seconds: float | None = None
    error: str | None = None

    @property
    def is_trap(self) -> bool:
        return bool(self.leak_terms)

    @property
    def hit(self) -> bool | None:
        """An expected page is among the passages sent to the LLM (None if not scored)."""
        if not self.expected_pages or self.should_refuse:
            return None
        return any(page_matches(r["page_title"], self.expected_pages) for r in self.retrieved)

    @property
    def retrieval_leaks(self) -> list[dict[str, Any]]:
        return [r for r in self.retrieved if r["spoiler_level"] > self.user_level]

    @property
    def leaked_terms(self) -> list[str]:
        if not self.answer:
            return []
        return [t for t in self.leak_terms if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", self.answer, re.I)]

    @property
    def refused(self) -> bool | None:
        if self.answer is None:
            return None
        return self.refused_before_llm or (self.citations == 0 and bool(REFUSAL_RE.search(self.answer)))


@dataclass
class Metric:
    name: str
    value: float | None
    target: str  # e.g. ">= 0.80", "" for informational metrics
    n: int  # denominator

    @property
    def passed(self) -> bool | None:
        if self.value is None or not self.target:
            return None
        op, bound = self.target.split()
        return {">=": self.value >= float(bound), "<=": self.value <= float(bound),
                "==": self.value == float(bound), "<": self.value < float(bound)}[op]


def _rate(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def score(results: list[Result]) -> list[Metric]:
    scored = [r.hit for r in results if r.hit is not None]
    metrics = [
        Metric("Recall@k", _rate(scored), ">= 0.80", len(scored)),
        Metric("Retrieval leak rate", _rate([bool(r.retrieval_leaks) for r in results]), "== 0", len(results)),
    ]
    answered = [r for r in results if r.answer is not None]
    if not answered:  # --retrieval-only
        return metrics
    traps = [r for r in answered if r.is_trap]
    refuse = [r for r in answered if r.should_refuse]
    answerable = [r for r in answered if not r.should_refuse and not r.is_trap]
    llm = [r for r in answered if not r.refused_before_llm]
    times = [r.seconds for r in answered if r.seconds is not None]
    return metrics + [
        Metric("Answer leak rate", _rate([bool(r.leaked_terms) for r in traps]), "<= 0.05", len(traps)),
        Metric("Refusal accuracy", _rate([bool(r.refused) for r in refuse]), ">= 0.90", len(refuse)),
        Metric("False refusals (answerable)", _rate([bool(r.refused) for r in answerable]), "", len(answerable)),
        Metric("Citation validity", _rate([not r.invalid_citations for r in llm]), "== 1", len(llm)),
        Metric("Median response time (s)", statistics.median(times) if times else None, "< 4", len(times)),
    ]


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def _excerpt(text: str | None, n: int = 240) -> str:
    text = (text or "").replace("\n", " ").replace("|", "\\|")
    return text if len(text) <= n else text[:n] + "…"


def render_report(results: list[Result], metrics: list[Metric], meta: dict[str, Any]) -> str:
    lines = [f"# Eval {meta['timestamp']}", ""]
    lines += [f"- **{k}:** {v}" for k, v in meta.items() if k != "timestamp"]
    lines += ["", "## Metrics", "", "| Metric | Value | Target | n | |", "|---|---|---|---|---|"]
    for m in metrics:
        mark = {True: "✅", False: "❌", None: ""}[m.passed]
        lines.append(f"| {m.name} | {_fmt(m.value)} | {m.target or '—'} | {m.n} | {mark} |")

    def section(title: str, rows: list[str], header: str) -> None:
        lines.extend(["", f"## {title} ({len(rows)})", ""])
        lines.extend([header, "|" + "---|" * (header.count("|") - 1), *rows] if rows else ["None."])

    section("Recall misses", [
        f"| {r.id} | L{r.user_level} | {r.question} | {', '.join(r.expected_pages)} | "
        f"{', '.join(dict.fromkeys(x['page_title'] for x in r.retrieved))} |"
        for r in results if r.hit is False], "| id | level | question | expected | retrieved pages |")
    section("Retrieval leaks", [
        f"| {r.id} | L{r.user_level} | {x['section_path']} (L{x['spoiler_level']}) |"
        for r in results for x in r.retrieval_leaks], "| id | level | chunk |")
    section("Answer leaks", [
        f"| {r.id} | L{r.user_level} | {r.question} | {', '.join(r.leaked_terms)} | {_excerpt(r.answer)} |"
        for r in results if r.leaked_terms], "| id | level | question | terms | answer |")
    section("Refusal errors", [
        f"| {r.id} | L{r.user_level} | {r.question} | {'should refuse' if r.should_refuse else 'wrongly refused'} "
        f"| {_excerpt(r.answer)} |"
        for r in results if r.refused is not None and not r.is_trap and bool(r.refused) != r.should_refuse],
        "| id | level | question | error | answer |")
    section("Invalid citations", [
        f"| {r.id} | {r.invalid_citations} |" for r in results if r.invalid_citations], "| id | removed |")
    section("Errors", [f"| {r.id} | {_excerpt(r.error)} |" for r in results if r.error], "| id | error |")
    return "\n".join(lines) + "\n"


def result_dict(r: Result) -> dict[str, Any]:
    return {**asdict(r), "hit": r.hit, "refused": r.refused, "leaked_terms": r.leaked_terms,
            "retrieval_leaks": len(r.retrieval_leaks)}
