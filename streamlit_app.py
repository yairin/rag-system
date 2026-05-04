"""
streamlit_app.py — ממשק Web לשאלות ותשובות על מסמכי משאבי אנוש
================================================================
הפעלה מקומית:  streamlit run streamlit_app.py
פרסום:         Streamlit Cloud (github + secrets)
"""

import os
# Force pure-Python protobuf — avoids C-extension crash on Python 3.14
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import streamlit as st

# ── הגדרת עמוד — חייב להיות ראשון ──────────────────────────────────────────
st.set_page_config(
    page_title="מערכת שאלות ותשובות — משאבי אנוש",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "מערכת RAG חכמה לשאילתות על מסמכי משאבי אנוש, מופעלת על ידי Claude AI.",
    },
)

# ── עיצוב RTL + עברית ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ─── RTL Global ─── */
    html, body, [class*="css"] { direction: rtl; }
    .stApp { direction: rtl; font-family: "Segoe UI", Arial, sans-serif; }

    /* ─── Sidebar ─── */
    [data-testid="stSidebar"] { direction: rtl; }
    [data-testid="stSidebar"] .stMarkdown { text-align: right; }

    /* ─── Chat messages ─── */
    [data-testid="stChatMessage"] { direction: rtl; text-align: right; }
    [data-testid="stChatInputContainer"] { direction: rtl; }
    [data-testid="stChatInputContainer"] textarea { direction: rtl; text-align: right; }

    /* ─── Cards for sources ─── */
    .source-card {
        background: #f0f4ff;
        border-right: 4px solid #4a6cf7;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.88rem;
        direction: rtl;
    }
    .source-score {
        display: inline-block;
        background: #4a6cf7;
        color: white;
        border-radius: 12px;
        padding: 1px 8px;
        font-size: 0.78rem;
        margin-right: 6px;
    }

    /* ─── Header ─── */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 20px 28px;
        border-radius: 12px;
        margin-bottom: 24px;
        text-align: right;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.7rem; }
    .main-header p  { color: #bfdbfe; margin: 4px 0 0; font-size: 0.95rem; }

    /* ─── Empty state ─── */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: #6b7280;
    }
    .empty-state .icon { font-size: 3rem; margin-bottom: 12px; }

    /* ─── Stats chips ─── */
    .stat-chip {
        display: inline-block;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e40af;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.82rem;
        margin: 2px;
    }

    /* ─── Expander fix ─── */
    [data-testid="stExpander"] { direction: rtl; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# טעינת מערכת ה-RAG (נטענת פעם אחת ונשמרת ב-cache)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⏳ טוען את מערכת ה-AI...")
def load_rag_system():
    """טוען את מערכת ה-RAG — נקרא פעם אחת בלבד."""
    from rag_system import RAGSystem

    # קרא API key מ-Streamlit Secrets או ממשתנה סביבה
    api_key = None
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        st.error(
            "🔑 מפתח ANTHROPIC_API_KEY לא הוגדר.\n\n"
            "הגדר אותו ב-Streamlit Cloud תחת Settings → Secrets:\n"
            "```\nANTHROPIC_API_KEY = \"sk-ant-...\"\n```"
        )
        st.stop()

    return RAGSystem(persist_dir="./rag_db", top_k=5, api_key=api_key)


# ══════════════════════════════════════════════════════════════════════════════
# אתחול Session State
# ══════════════════════════════════════════════════════════════════════════════

def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []          # היסטוריית שיחה
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(rag):
    with st.sidebar:
        st.markdown("## 📚 מסמכי הידע")

        sources = rag.list_sources()
        stats = rag.stats()

        if not sources:
            st.warning("⚠️ אין מסמכים טעונים.\nהרץ `build_db.py` תחילה.")
        else:
            st.markdown(
                f'<span class="stat-chip">📄 {len(sources)} מסמכים</span>'
                f'<span class="stat-chip">🧩 {stats["total_chunks"]:,} קטעים</span>',
                unsafe_allow_html=True,
            )
            st.markdown("---")
            for src in sources:
                # הצג שם מסמך נקי
                display = src if len(src) <= 45 else src[:42] + "..."
                st.markdown(f"📄 {display}")

        st.markdown("---")
        st.markdown("### ⚙️ הגדרות שיחה")

        top_k = st.slider(
            "כמות קטעים לחיפוש",
            min_value=2, max_value=10, value=5,
            help="כמה קטעי מסמך לשלוף עבור כל שאלה"
        )

        min_score = st.slider(
            "סף רלוונטיות מינימלי",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            help="סנן קטעים עם ציון נמוך מדי (0 = ללא סינון)"
        )

        show_sources = st.toggle("הצג מקורות לכל תשובה", value=True)
        use_history = st.toggle("זכור היסטוריית שיחה", value=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ נקה שיחה", use_container_width=True):
                st.session_state.messages = []
                rag.clear_conversation()
                st.rerun()
        with col2:
            if st.button("ℹ️ סטטיסטיקות", use_container_width=True):
                st.info(
                    f"**מודל:** {stats['llm_model']}\n\n"
                    f"**Embeddings:** {stats['embed_model']}\n\n"
                    f"**שאילתות:** {st.session_state.total_queries}\n\n"
                    f"**Tokens:** {st.session_state.total_tokens:,}"
                )

        return top_k, min_score, show_sources, use_history


# ══════════════════════════════════════════════════════════════════════════════
# הצגת מקורות
# ══════════════════════════════════════════════════════════════════════════════

def render_sources(sources):
    if not sources:
        return

    with st.expander(f"📎 מקורות ({len(sources)} קטעים)", expanded=False):
        for i, src in enumerate(sources, 1):
            page_info = f" | עמוד {src.page}" if src.page else ""
            score_pct = int(src.score * 100)
            score_color = "#16a34a" if score_pct >= 70 else "#ca8a04" if score_pct >= 40 else "#dc2626"

            st.markdown(
                f"""<div class="source-card">
                    <strong>מקור {i}: {src.source}{page_info}</strong>
                    <span class="source-score" style="background:{score_color}">
                        {score_pct}% התאמה
                    </span>
                    <br><span style="color:#374151; font-size:0.85rem;">
                        {src.text[:280].strip()}{'...' if len(src.text) > 280 else ''}
                    </span>
                </div>""",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# ממשק ראשי
# ══════════════════════════════════════════════════════════════════════════════

def main():
    init_session()
    rag = load_rag_system()
    top_k, min_score, show_sources, use_history = render_sidebar(rag)

    # ── כותרת ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>📋 מערכת שאלות ותשובות — משאבי אנוש</h1>
        <p>שאל כל שאלה הנוגעת לתנאי העסקה, שכר, פרישה, משמעת ועוד — המערכת תחפש בקרב המסמכים ותענה בדייקנות.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── הצג שאלות לדוגמה אם שיחה ריקה ────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("#### 💡 שאלות לדוגמה")
        example_questions = [
            "מהם תנאי הקבלה לעבודה?",
            "כיצד מחושב השכר הבסיסי?",
            "מהם תנאי הפרישה מהעבודה?",
            "מה הן ההוראות בנוגע למשמעת עובדים?",
            "אילו השתלמויות זכאי עובד לקבל?",
            "מה הם ההסדרים עם האיגודים המקצועיים?",
        ]
        cols = st.columns(3)
        for i, q in enumerate(example_questions):
            if cols[i % 3].button(q, use_container_width=True, key=f"example_{i}"):
                st.session_state["pending_question"] = q
                st.rerun()

    # ── הצג היסטוריית שיחה ────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and show_sources and "sources" in msg:
                render_sources(msg["sources"])
            if msg["role"] == "assistant" and "tokens" in msg:
                st.caption(f"🔢 {msg['tokens']:,} tokens")

    # ── שאלה ממתינה (מלחצן דוגמה) ────────────────────────────────────────────
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")
        _process_question(rag, question, top_k, min_score, show_sources, use_history)

    # ── קלט משתמש ─────────────────────────────────────────────────────────────
    if question := st.chat_input("✍️ שאל שאלה על מסמכי משאבי האנוש..."):
        _process_question(rag, question, top_k, min_score, show_sources, use_history)


def _process_question(rag, question, top_k, min_score, show_sources, use_history):
    """מעבד שאלה ומציג תשובה."""
    # הצג שאלת משתמש
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # ייצר תשובה
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 מחפש במסמכים ומייצר תשובה..."):
            try:
                response = rag.query(
                    question=question,
                    top_k=top_k,
                    use_history=use_history,
                    min_score=min_score,
                )
            except Exception as e:
                st.error(f"שגיאה: {e}")
                return

        st.markdown(response.answer)

        if show_sources:
            render_sources(response.sources)

        if response.tokens_used:
            st.caption(f"🔢 {response.tokens_used:,} tokens")

    # שמור להיסטוריה
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "sources": response.sources,
        "tokens": response.tokens_used,
    })
    st.session_state.total_queries += 1
    st.session_state.total_tokens += response.tokens_used


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
