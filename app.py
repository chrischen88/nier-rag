"""Streamlit chat UI (SPEC §10): `uv run streamlit run app.py`.

Each question is answered on its own: chat history isn't sent to the model (query rewriting for
follow-ups is on the roadmap, SPEC §14). Turns asked at a later progress level than the current one
are hidden, so lowering the slider also hides answers that could now be spoilers.

When deployed (Dockerfile, fly.toml), two guards limit OpenAI spend: `app.daily_question_limit` in
config.yaml caps questions per server process per day, and setting the APP_PASSWORD environment variable
asks visitors for a password.
"""

from __future__ import annotations

import dataclasses
import hmac
import os
import threading
from dataclasses import dataclass, field
from datetime import date

import streamlit as st

from src.config import ConfigError, load_config, resolve_path
from src.generate import Answer, Assistant
from src.index import IndexMismatchError, get_collection, open_client
from src.providers import ProviderError, get_embedder, get_llm, get_moderator
from src.providers.openai import Usage, estimate_cost
from src.spoilers import PROGRESS_LEVELS


@dataclass
class Turn:
    question: str
    level: int
    answer: Answer | None = None
    error: str | None = None


@dataclass
class DailyLimit:
    """Questions allowed per day, shared by every session in this server process (0 = unlimited)."""
    limit: int
    day: date | None = None
    count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def take(self) -> bool:
        with self._lock:
            if self.day != date.today():
                self.day, self.count = date.today(), 0
            if self.limit and self.count >= self.limit:
                return False
            self.count += 1
            return True


@st.cache_resource
def daily_limit(limit: int) -> DailyLimit:
    return DailyLimit(limit)


@st.cache_resource(show_spinner="Opening the archive…")
def load_assistant() -> Assistant:
    cfg = load_config()
    embedder, llm = get_embedder(cfg), get_llm(cfg)
    return Assistant(cfg, get_collection(open_client(resolve_path(cfg, "chroma")), embedder), embedder, llm,
                     get_moderator(cfg))


def md(text: str) -> str:
    return text.replace("$", r"\$")  # Streamlit renders $...$ as LaTeX


def usage_snapshot(a: Assistant) -> tuple[int, int, int]:
    llm, emb = getattr(a.llm, "usage", Usage()), getattr(a.embedder, "usage", Usage())
    return llm.prompt_tokens, llm.completion_tokens, emb.embedding_tokens


def render_answer(ans: Answer, debug: bool) -> None:
    st.markdown(md(ans.text))
    for c in ans.citations:
        with st.expander(f"[{c.n}] {c.section_path}"):
            st.markdown(md(ans.passages[c.n - 1].text.split("\n\n", 1)[-1]))
            st.markdown(f"[Open on the NieR Wiki]({c.url})")
    if debug:
        render_debug(ans)


def render_debug(ans: Answer) -> None:
    with st.expander("Debug"):
        how = "refused before calling the LLM" if ans.refused_before_llm else f"{len(ans.citations)} cited"
        if ans.guardrail:
            how += f" · guardrail: {ans.guardrail}"
        st.caption(f"{ans.seconds:.2f}s · {len(ans.passages)} passages retrieved · {how}")
        if ans.invalid_citations:
            st.warning(f"Removed citations to passages that don't exist: {ans.invalid_citations}")
        if ans.passages:
            st.dataframe([{"#": i, "similarity": round(p.similarity, 3), "level": p.metadata["spoiler_level"],
                           "type": p.metadata["content_type"], "section": p.metadata["section_path"]}
                          for i, p in enumerate(ans.passages, 1)], hide_index=True)
        for m in ans.messages or []:
            st.caption(m["role"])
            st.code(m["content"], language=None, wrap_lines=True)


def render_turn(turn: Turn, debug: bool) -> None:
    with st.chat_message("user"):
        st.markdown(md(turn.question))
    with st.chat_message("assistant"):
        if turn.answer is not None:
            render_answer(turn.answer, debug)
        if turn.error is not None:
            st.error(turn.error)


def answer_question(a: Assistant, question: str, level: int, hide_speculation: bool, debug: bool) -> Turn:
    """Stream an answer into the current container, then redraw it with resolved citations."""
    turn = Turn(question, level)
    slot = st.empty()
    try:
        plan = a.prepare(question, level, hide_speculation=hide_speculation)
        raw = slot.write_stream(a.stream(plan))
        turn.answer = a.finish(plan, raw if isinstance(raw, str) else "".join(map(str, raw)))
    except ProviderError as e:
        turn.error = f"{e}\n\nTry again in a moment."
        slot.empty()
        st.error(turn.error)
        return turn
    # Drop the stored embeddings (several KB per passage) before keeping the turn in session state.
    turn.answer.passages = [dataclasses.replace(p, embedding=None) for p in turn.answer.passages]
    with slot.container():
        render_answer(turn.answer, debug)
    return turn


st.set_page_config(page_title="YoRHa Archive", page_icon="📜")
state = st.session_state
state.setdefault("turns", [])
state.setdefault("tokens", (0, 0, 0))  # prompt, completion, embedding tokens this session

with st.sidebar:
    level = st.select_slider(
        "How far have you played?",
        options=list(PROGRESS_LEVELS),
        format_func=lambda lv: PROGRESS_LEVELS[lv][0],
        key="progress_level",
    )
    st.caption(f"Unlocked: {PROGRESS_LEVELS[level][1]}")
    hide_speculation = st.toggle("Hide fan speculation", key="hide_speculation")
    debug = st.toggle("Debug panel", key="debug")
    models_slot = st.empty()
    cost_slot = st.empty()
    st.info("Questions and retrieved wiki passages are sent to OpenAI.")
    if st.button("Clear chat"):
        state.turns = []

st.title("YoRHa Archive")

if (password := os.environ.get("APP_PASSWORD")) and not state.get("authed"):
    with st.form("login"):
        given = st.text_input("Password", type="password")
        if st.form_submit_button("Enter"):
            if hmac.compare_digest(given.encode(), password.encode()):
                state.authed = True
                st.rerun()
            st.error("Wrong password.")
    st.stop()

try:
    assistant = load_assistant()
except (ConfigError, IndexMismatchError) as e:
    st.error(str(e))
    st.stop()

cfg = assistant.cfg
models_slot.caption(f"Chat model: `{cfg['llm']['model']}`  \nEmbedder: `{cfg['embeddings']['model']}`  \n"
                    f"Index: `{assistant.col.name}` ({assistant.col.count():,} chunks)  \n"
                    "Changing the embedder requires `scripts/ingest.py --reindex`.")

visible = [t for t in state.turns if t.level <= level]
if hidden := len(state.turns) - len(visible):
    st.caption(f"{hidden} earlier question{'s' if hidden > 1 else ''} hidden: asked at a later progress level.")
for turn in visible:
    render_turn(turn, debug)

if question := st.chat_input("Ask about NieR:Automata lore",
                             max_chars=cfg.get("guardrails", {}).get("max_question_chars")):
    with st.chat_message("user"):
        st.markdown(md(question))
    with st.chat_message("assistant"):
        if not daily_limit(cfg.get("app", {}).get("daily_question_limit", 0)).take():
            state.turns.append(Turn(question, level, error="The archive has answered all the questions it can "
                                                           "for today. Please come back tomorrow."))
            st.error(state.turns[-1].error)
        else:
            before = usage_snapshot(assistant)
            state.turns.append(answer_question(assistant, question, level, hide_speculation, debug))
            state.tokens = tuple(t + a - b for t, a, b in zip(state.tokens, usage_snapshot(assistant), before))

prompt_toks, completion_toks, embed_toks = state.tokens
cost = estimate_cost(cfg, Usage(prompt_tokens=prompt_toks, completion_tokens=completion_toks),
                     Usage(embedding_tokens=embed_toks))
cost_slot.caption(md(f"This session: {prompt_toks + completion_toks:,} chat tokens, "
                     f"{embed_toks:,} embedding tokens · ~${cost:.4f}"))

st.caption("Lore from the [NieR Wiki](https://nier.fandom.com) on Fandom, licensed under "
           "[CC BY-SA](https://creativecommons.org/licenses/by-sa/3.0/). Answers cite their sources.")
