"""
Step 7 — Streamlit Chat Interface

Run with:
    .venv\\Scripts\\streamlit.exe run ui/streamlit_ui.py

Features:
  • Persistent chat history using st.session_state
  • Bubble-style messages (user on right, bot on left)
  • Confidence score shown under each bot reply
  • Live log viewer for unknown questions (sidebar)
"""

import os
import sys

# Ensure the project root is on sys.path so `chatbot` package is importable
# regardless of which directory Streamlit is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from chatbot.faq_engine import FAQEngine, UNKNOWN_LOG

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="University FAQ Chatbot",
    page_icon="🎓",
    layout="centered",
)

# ---------------------------------------------------------------------------
# CSS — simple chat bubbles
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.user-bubble {
    background-color: #DCF8C6;
    border-radius: 12px 12px 2px 12px;
    padding: 10px 14px;
    margin: 6px 0 6px 20%;
    text-align: right;
    color: #000;
}
.bot-bubble {
    background-color: #F1F0F0;
    border-radius: 12px 12px 12px 2px;
    padding: 10px 14px;
    margin: 6px 20% 6px 0;
    color: #000;
}
.score-tag {
    font-size: 0.75em;
    color: #888;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Initialise the FAQ engine once and store in session state
# ---------------------------------------------------------------------------
@st.cache_resource
def load_engine() -> FAQEngine:
    return FAQEngine()


engine = load_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role": "user"|"bot", "content": str, "meta": dict|None}

# ---------------------------------------------------------------------------
# Sidebar — unknown questions log
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🔍 Unknown Questions Log")
    st.caption("Questions the chatbot couldn't answer confidently.")

    if os.path.exists(UNKNOWN_LOG) and os.path.getsize(UNKNOWN_LOG) > 0:
        log_df = pd.read_csv(UNKNOWN_LOG)
        if log_df.empty:
            st.info("No unknown questions logged yet.")
        else:
            st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info("No unknown questions logged yet.")

    if st.button("🗑️ Clear log"):
        with open(UNKNOWN_LOG, "w", newline="", encoding="utf-8") as f:
            import csv
            csv.writer(f).writerow(["timestamp", "question", "best_score"])
        st.rerun()

    st.divider()
    st.caption("Confidence threshold: **0.40**")
    st.caption("Matching method: **TF-IDF + Cosine Similarity**")

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("🎓 University FAQ Chatbot")
st.caption("Ask me anything about admissions, fees, programs, and more.")

# Render existing conversation history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">🧑‍🎓 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        meta = msg.get("meta", {})
        score_html = ""
        if meta:
            icon  = "✅" if meta.get("confident") else "⚠️"
            score_html = (
                f'<div class="score-tag">{icon} Confidence: {meta["score"]:.2%} '
                f'— Matched: <em>{meta["matched_question"]}</em></div>'
            )
        st.markdown(
            f'<div class="bot-bubble">🤖 {msg["content"]}{score_html}</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_input = st.chat_input("Type your question here…")

if user_input:
    user_input = user_input.strip()
    if user_input:
        # Store and display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(
            f'<div class="user-bubble">🧑‍🎓 {user_input}</div>',
            unsafe_allow_html=True,
        )

        # Get answer from engine
        result = engine.ask(user_input)

        # Store and display bot reply
        st.session_state.messages.append({
            "role":    "bot",
            "content": result["answer"],
            "meta":    result,
        })

        icon  = "✅" if result["confident"] else "⚠️"
        score_html = (
            f'<div class="score-tag">{icon} Confidence: {result["score"]:.2%} '
            f'— Matched: <em>{result["matched_question"]}</em></div>'
        )
        st.markdown(
            f'<div class="bot-bubble">🤖 {result["answer"]}{score_html}</div>',
            unsafe_allow_html=True,
        )

        st.rerun()

