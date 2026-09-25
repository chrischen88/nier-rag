"""Streamlit entry point (SPEC §10). Placeholder until M5: `uv run streamlit run app.py`."""

import streamlit as st

from src.config import load_config
from src.spoilers import PROGRESS_LEVELS

cfg = load_config()
st.set_page_config(page_title="YoRHa Archive", page_icon="📜")

with st.sidebar:
    level = st.select_slider(
        "How far have you played?",
        options=list(PROGRESS_LEVELS),
        format_func=lambda lv: PROGRESS_LEVELS[lv][0],
        key="progress_level",
    )
    st.toggle("Hide fan speculation", key="hide_speculation")
    st.caption(f"Chat model: `{cfg['llm']['model']}`  \nEmbedder: `{cfg['embeddings']['model']}`")
    st.info("Questions and retrieved wiki passages are sent to OpenAI.")

st.title("YoRHa Archive")
st.write("Chat arrives in M5.")
st.caption("Lore from the [NieR Wiki](https://nier.fandom.com) on Fandom, licensed under CC BY-SA.")
