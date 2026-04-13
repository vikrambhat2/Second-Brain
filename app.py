import os
import datetime
import uuid
import logging
from typing import Dict, Optional
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct, VectorParams, Distance,
    Filter, FieldCondition, MatchValue,
    PayloadSchemaType, Range
)
import PyPDF2
import docx
from bs4 import BeautifulSoup
import io

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Second Brain",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 14px;
}

/* ── Feed entry cards ── */
.entry-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 10px;
    line-height: 1.6;
}
.entry-time {
    font-size: 11px;
    color: #8b949e;
    font-family: 'Courier New', monospace;
    margin-bottom: 4px;
}
.entry-label {
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 6px;
    color: #e6edf3;
}
.entry-text {
    font-size: 14px;
    color: #b0bec5;
    line-height: 1.55;
}
.cat-pill {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    margin-left: 8px;
    vertical-align: middle;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: rgba(108,99,255,0.08);
    border: 1px solid rgba(108,99,255,0.18);
    border-radius: 10px;
    padding: 10px 14px;
}

/* ── Primary button glow ── */
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #6c63ff, #5a52d5);
    border: none;
    box-shadow: 0 0 16px rgba(108,99,255,0.35);
    font-weight: 600;
    letter-spacing: 0.3px;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── Login card ── */
.login-wrap {
    max-width: 380px;
    margin: 80px auto 0;
    text-align: center;
}
.login-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 4px;
}
.login-sub {
    font-size: 15px;
    color: #8b949e;
    margin-bottom: 32px;
}
</style>
""", unsafe_allow_html=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')]
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CATEGORIES = [
    "Journal", "Personal Life", "Family", "Health & Fitness",
    "Finances", "Work Projects", "Career & Goals", "Ideas & Inspiration",
    "Travel Plans", "Shopping Lists", "Daily Tasks", "House Building",
    "Random Thoughts",
]

# Color for each category (used in feed pills)
CAT_COLOR = {
    "Journal":            ("#ff6b6b", "#3d1a1a"),
    "Personal Life":      ("#ffd166", "#3d3010"),
    "Family":             ("#06d6a0", "#0a2e25"),
    "Health & Fitness":   ("#4cc9f0", "#0d2b38"),
    "Finances":           ("#f9c74f", "#3d3010"),
    "Work Projects":      ("#a78bfa", "#251d3d"),
    "Career & Goals":     ("#f472b6", "#3d1030"),
    "Ideas & Inspiration":("#38bdf8", "#0d2535"),
    "Travel Plans":       ("#34d399", "#0a2e20"),
    "Shopping Lists":     ("#fb923c", "#3d1e0a"),
    "Daily Tasks":        ("#a3e635", "#1e2e0a"),
    "House Building":     ("#fb923c", "#3d1e0a"),
    "Random Thoughts":    ("#c084fc", "#251535"),
}

QUICK_LOG_TYPES = [
    "💭 Free Thought",
    "✅ Task",
    "🔔 Reminder",
    "🏠 House Building",
    "👨‍👩‍👧 Family",
    "💰 Expense",
    "💼 Work Session",
    "💡 Idea",
    "🍽️ Meal",
    "🏃 Exercise",
]

QUICK_LOG_CATEGORY = {
    "💭 Free Thought":   "Random Thoughts",
    "✅ Task":           "Daily Tasks",
    "🔔 Reminder":       "Daily Tasks",
    "🏠 House Building": "House Building",
    "👨‍👩‍👧 Family":        "Family",
    "💰 Expense":        "Finances",
    "💼 Work Session":   "Work Projects",
    "💡 Idea":           "Ideas & Inspiration",
    "🍽️ Meal":          "Health & Fitness",
    "🏃 Exercise":       "Health & Fitness",
}

# ── Env & clients ─────────────────────────────────────────────────────────────
load_dotenv()

try:
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_CLOUD_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    COLLECTION_NAME = "second-brain-index"
except Exception as e:
    logger.error(f"Initialization error: {e}")
    st.error("Failed to initialize components. Check credentials and dependencies.")
    st.stop()

# ── Qdrant setup ──────────────────────────────────────────────────────────────
def init_qdrant():
    try:
        if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        for field, schema in [
            ("category",       PayloadSchemaType.KEYWORD),
            ("text",           PayloadSchemaType.TEXT),
            ("timestamp",      PayloadSchemaType.KEYWORD),
            ("timestamp_unix", PayloadSchemaType.INTEGER),
        ]:
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=schema
            )
    except Exception as e:
        logger.error(f"Qdrant init error: {e}")
        st.error("Qdrant initialization failed.")
        st.stop()

init_qdrant()

# ── Auth ──────────────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("""
    <div class="login-wrap">
        <div class="login-title">🧠</div>
        <div class="login-title" style="font-size:28px;margin-top:-8px;">Second Brain</div>
        <div class="login-sub">Your personal memory assistant</div>
    </div>
    """, unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        password = st.text_input("", type="password", placeholder="Enter password...",
                                 label_visibility="collapsed")
        if st.button("Unlock", use_container_width=True, type="primary"):
            if password == os.getenv("APP_PASSWORD", ""):
                st.session_state["authenticated"] = True
                st.rerun()
            elif password:
                st.error("Incorrect password")
    st.stop()

# ── Core logic ────────────────────────────────────────────────────────────────

def add_note(text: str, category: str, extra_payload: dict = None) -> Dict:
    try:
        if not text or not text.strip():
            return {"status": "error", "message": "Empty note"}
        now = datetime.datetime.utcnow()
        note_id = str(uuid.uuid4())
        vector = embedder.embed_query(text)
        payload = {
            "text": text,
            "timestamp": now.isoformat(),
            "timestamp_unix": int(now.timestamp()),
            "category": category,
        }
        if extra_payload:
            payload.update(extra_payload)
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=note_id, vector=vector, payload=payload)]
        )
        return {"status": "success", "note_id": note_id}
    except Exception as e:
        logger.error(f"Error storing note: {e}")
        return {"status": "error", "message": str(e)}


def get_today_entries() -> list:
    try:
        today_start = datetime.datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        today_unix = int(today_start.timestamp())
        results, offset = [], None
        while True:
            batch, next_offset = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                scroll_filter=Filter(
                    must=[FieldCondition(key="timestamp_unix", range=Range(gte=today_unix))]
                )
            )
            results.extend(batch)
            if next_offset is None:
                break
            offset = next_offset
        return sorted(results, key=lambda x: x.payload.get("timestamp", ""), reverse=True)
    except Exception as e:
        logger.error(f"Error fetching today's entries: {e}")
        return []


def extract_text(content: bytes, source: str) -> str:
    try:
        buf = io.BytesIO(content)
        if source.endswith(".txt"):
            return buf.read().decode("utf-8")
        elif source.endswith(".pdf"):
            reader = PyPDF2.PdfReader(buf)
            return " ".join(p.extract_text() for p in reader.pages if p.extract_text())
        elif source.endswith(".docx"):
            doc = docx.Document(buf)
            return " ".join(p.text for p in doc.paragraphs)
        elif source.endswith(".html"):
            return BeautifulSoup(buf.read(), "html.parser").get_text()
        return ""
    except Exception as e:
        logger.error(f"Extraction error for {source}: {e}")
        return ""


def query_notes(question: str, selected_categories: Optional[list] = None, top_k: int = 5) -> Dict:
    try:
        query_vector = embedder.embed_query(question)
        filters = None
        if selected_categories:
            filters = Filter(
                should=[
                    FieldCondition(key="category", match=MatchValue(value=cat))
                    for cat in selected_categories
                ]
            )
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=filters
        )
        if not results:
            return {"answer": "No relevant information found.", "results": []}

        context = "\n\n".join([
            f"{r.payload.get('timestamp', '')} [{r.payload.get('category', 'Uncategorized')}]: {r.payload.get('text', '')}"
            for r in results
        ])
        today_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        messages = [
            SystemMessage(content=(
                f"Today is {today_str}.\n"
                "You are a personal memory assistant. Answer only based on the notes provided. "
                "Be clear and concise. If there is not enough information, say so."
            )),
            HumanMessage(content=(
                f"Question: {question}\n\nRelevant notes:\n{context}\n\nAnswer based only on these notes."
            ))
        ]
        response = llm.invoke(messages)
        return {
            "answer": response.content,
            "results": [
                {"category": r.payload.get("category", ""),
                 "timestamp": r.payload.get("timestamp", ""),
                 "text": r.payload.get("text", "")}
                for r in results
            ]
        }
    except Exception as e:
        logger.error(f"Query error: {e}")
        return {"answer": f"Error: {str(e)}", "results": []}


# ── Helper: category pill HTML ────────────────────────────────────────────────
def cat_pill(cat: str) -> str:
    fg, bg = CAT_COLOR.get(cat, ("#aaa", "#222"))
    return (
        f'<span class="cat-pill" style="background:{bg};color:{fg};'
        f'border:1px solid {fg}33;">{cat}</span>'
    )


# ── Session state defaults ────────────────────────────────────────────────────
if "chat_history"    not in st.session_state: st.session_state.chat_history    = []
if "log_type_idx" not in st.session_state or st.session_state.log_type_idx >= len(QUICK_LOG_TYPES):
    st.session_state.log_type_idx = 0
if "category_filters" not in st.session_state:
    st.session_state.category_filters = {c: False for c in CATEGORIES}

# ── App header ────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='margin-bottom:0;font-weight:800;'>🧠 Second Brain</h2>",
    unsafe_allow_html=True
)
now_str = datetime.datetime.utcnow().strftime("%A, %B %d")
st.caption(f"Today is {now_str}")

quick_tab, store_tab, feed_tab, chat_tab = st.tabs([
    "⚡ Quick Log", "📥 Store", "📅 Today", "💬 Chat"
])


# ══════════════════════════════════════════════════════════════════════════════
# ⚡  QUICK LOG
# ══════════════════════════════════════════════════════════════════════════════
with quick_tab:

    # ── Log-type picker (3×3 grid of buttons) ─────────────────────────────────
    st.markdown("**What are you logging?**")
    rows = [QUICK_LOG_TYPES[i:i+3] for i in range(0, len(QUICK_LOG_TYPES), 3)]
    for row in rows:
        cols = st.columns(3)
        for col, lt in zip(cols, row):
            idx = QUICK_LOG_TYPES.index(lt)
            is_selected = st.session_state.log_type_idx == idx
            if col.button(
                lt,
                use_container_width=True,
                type="primary" if is_selected else "secondary",
                key=f"lt_btn_{idx}"
            ):
                st.session_state.log_type_idx = idx
                st.rerun()

    st.divider()

    log_type = QUICK_LOG_TYPES[st.session_state.log_type_idx]
    category = QUICK_LOG_CATEGORY[log_type]
    note_text = ""
    extra_payload = {"log_type": log_type}

    # ── Forms per log type ────────────────────────────────────────────────────

    if log_type == "💭 Free Thought":
        note_text = st.text_area(
            "What's on your mind?", height=150,
            placeholder="Just type...", label_visibility="collapsed"
        )
        category = st.selectbox("Category", CATEGORIES,
                                index=CATEGORIES.index(category))

    elif log_type == "✅ Task":
        task = st.text_input("Task", placeholder="What needs to be done?")
        col1, col2 = st.columns(2)
        status   = col1.selectbox("Status", ["To Do", "In Progress", "Done", "Blocked"])
        priority = col2.selectbox("Priority", ["Low", "Medium", "High", "Urgent"])
        note = st.text_input("Notes (optional)", placeholder="Context, next steps...")
        note_text = (
            f"Task: {task} — {status}, {priority} priority."
            f"{(' ' + note) if note else ''}"
        ).strip()
        extra_payload.update({"task_status": status, "priority": priority})

    elif log_type == "🔔 Reminder":
        what = st.text_input("What to remember", placeholder="Call doctor, submit form, follow up with...")
        when = st.text_input("When", placeholder="Tomorrow, Friday, next week, 3pm...")
        note = st.text_input("Notes (optional)")
        note_text = (
            f"Reminder: {what}."
            f"{(' When: ' + when) if when else ''}"
            f"{(' ' + note) if note else ''}"
        ).strip()
        extra_payload.update({"reminder_when": when})

    elif log_type == "🏠 House Building":
        area = st.text_input("Area / topic", placeholder="Foundation, plumbing, electrical, interior...")
        update = st.text_area("Update / progress", height=100,
                              placeholder="What happened? What was decided?")
        col1, col2 = st.columns(2)
        issues = col1.text_input("Issues / blockers (optional)")
        cost   = col2.number_input("Cost incurred (₹)", min_value=0.0, step=100.0, format="%.0f")
        note_text = (
            f"House building — {area}: {update}."
            f"{(' Issues: ' + issues) if issues else ''}"
            f"{(f' Cost: ₹{cost:.0f}') if cost > 0 else ''}"
        ).strip()
        extra_payload.update({"area": area, "cost": cost if cost > 0 else None})

    elif log_type == "👨‍👩‍👧 Family":
        person = st.text_input("Who?", placeholder="Name or relationship (mum, dad, sister...)")
        what   = st.text_area("What happened?", height=100,
                              placeholder="Topic discussed, event, update...")
        note   = st.text_input("Notes (optional)")
        note_text = (
            f"Family — {person}: {what}."
            f"{(' ' + note) if note else ''}"
        ).strip()
        extra_payload.update({"person": person})

    elif log_type == "🏃 Exercise":
        activity = st.text_input("Activity", placeholder="Running, gym, yoga, swim...")
        col1, col2 = st.columns(2)
        duration  = col1.number_input("Duration (mins)", 0, 480, 30, 5)
        intensity = col2.selectbox("Intensity", ["Low","Moderate","High","Max"])
        note = st.text_area("Notes (optional)", height=80,
                            placeholder="How did it feel?")
        note_text = f"Exercise: {activity}, {duration} mins, {intensity.lower()} intensity. {note}".strip(" .")
        extra_payload.update({"activity": activity, "duration_mins": duration, "intensity": intensity})

    elif log_type == "💰 Expense":
        col1, col2 = st.columns(2)
        amount  = col1.number_input("Amount ($)", 0.0, step=0.01, format="%.2f")
        payment = col2.selectbox("Via", ["Cash","Credit Card","Debit Card","UPI","Other"])
        desc    = st.text_input("What for?", placeholder="Coffee, groceries, subscription...")
        note    = st.text_input("Notes (optional)")
        note_text = f"Expense: ${amount:.2f} via {payment} — {desc}. {note}".strip(" .")
        extra_payload.update({"amount": amount, "payment_method": payment})

    elif log_type == "🍽️ Meal":
        meal_type = st.selectbox("Meal", ["Breakfast","Lunch","Dinner","Snack"])
        food      = st.text_area("What did you eat?", height=80,
                                 placeholder="Be as detailed as you like...")
        col1, col2 = st.columns(2)
        hunger = col1.slider("Hunger before (1–5)", 1, 5, 3)
        satis  = col2.slider("Satisfaction after (1–5)", 1, 5, 4)
        note   = st.text_input("Notes (optional)", placeholder="Taste? Where were you?")
        note_text = (
            f"{meal_type}: {food}. "
            f"Hunger before {hunger}/5, satisfaction {satis}/5. {note}"
        ).strip(" .")
        extra_payload.update({"meal_type": meal_type, "hunger_before": hunger, "satisfaction": satis})

    elif log_type == "💼 Work Session":
        focus    = st.text_input("Focus area / project",
                                 placeholder="What were you working on?")
        duration = st.number_input("Duration (mins)", 0, 480, 60, 15)
        done     = st.text_area("What did you accomplish?", height=80,
                                placeholder="Key outputs, decisions, progress...")
        blockers = st.text_input("Blockers (optional)",
                                 placeholder="Anything that slowed you down?")
        note_text = (
            f"Work session on '{focus}', {duration} mins. "
            f"Accomplished: {done}."
            f"{(' Blockers: ' + blockers) if blockers else ''}"
        ).strip()
        extra_payload.update({"focus_area": focus, "duration_mins": duration})

    elif log_type == "💡 Idea":
        title    = st.text_input("Idea title", placeholder="Give it a name...")
        desc     = st.text_area("Describe the idea", height=120,
                                placeholder="What is it? Why does it matter?")
        context  = st.text_input("What triggered it? (optional)")
        note_text = (
            f"Idea — {title}: {desc}."
            f"{(' Context: ' + context) if context else ''}"
        ).strip()

    st.divider()
    if st.button("⚡  Log It", use_container_width=True, type="primary"):
        if note_text.strip():
            with st.spinner("Saving..."):
                result = add_note(text=note_text, category=category,
                                  extra_payload=extra_payload)
            if result["status"] == "success":
                st.success("Logged!")
                st.balloons()
            else:
                st.error(f"Error: {result['message']}")
        else:
            st.warning("Nothing to log — fill in the fields first.")


# ══════════════════════════════════════════════════════════════════════════════
# 📥  STORE
# ══════════════════════════════════════════════════════════════════════════════
with store_tab:
    st.markdown("### Store a note or document")

    with st.expander("Select a category"):
        st.caption("Pick exactly one")
        store_cols = st.columns(3)
        store_selected = []
        for idx, cat in enumerate(CATEGORIES):
            if store_cols[idx % 3].checkbox(cat, key=f"store_cat_{cat}"):
                store_selected.append(cat)

    text_input = st.text_area("Paste or type your notes here", height=200)

    with st.expander("Or upload a file"):
        uploaded_file = st.file_uploader(
            "PDF, DOCX, TXT or HTML", type=["pdf","docx","txt","html"]
        )

    if st.button("Store Memory", use_container_width=True, type="primary"):
        with st.spinner("Processing..."):
            try:
                if len(store_selected) != 1:
                    st.warning("Please select exactly one category.")
                    st.stop()
                category = store_selected[0]

                if text_input:
                    result = add_note(text=text_input, category=category)
                    if result["status"] == "success":
                        st.success("Text stored!")
                    else:
                        st.error(f"Error: {result['message']}")

                if uploaded_file:
                    text = extract_text(uploaded_file.read(), uploaded_file.name)
                    if text:
                        result = add_note(text=text, category=category)
                        if result["status"] == "success":
                            st.success(f"Stored {uploaded_file.name}!")
                        else:
                            st.error(f"Error: {result['message']}")
                    else:
                        st.error(f"Could not extract text from {uploaded_file.name}.")
            except Exception as e:
                st.error(f"Error: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# 📅  TODAY'S FEED
# ══════════════════════════════════════════════════════════════════════════════
with feed_tab:
    col_h, col_r = st.columns([5, 1])
    col_h.markdown(
        f"### {datetime.datetime.utcnow().strftime('%A, %B %d')}"
    )
    if col_r.button("🔄", help="Refresh"):
        st.rerun()

    with st.spinner("Loading..."):
        entries = get_today_entries()

    if not entries:
        st.markdown(
            "<div style='text-align:center;color:#8b949e;padding:60px 0;font-size:15px;'>"
            "No entries yet today.<br>Tap <b>⚡ Quick Log</b> to start.</div>",
            unsafe_allow_html=True
        )
    else:
        # Summary row
        cat_counts: dict = {}
        for e in entries:
            c = e.payload.get("category", "Uncategorized")
            cat_counts[c] = cat_counts.get(c, 0) + 1

        num_cols = min(len(cat_counts), 4)
        m_cols = st.columns(num_cols)
        for i, (c, n) in enumerate(cat_counts.items()):
            m_cols[i % num_cols].metric(c, n)

        st.divider()

        # Timeline cards
        for entry in entries:
            ts = entry.payload.get("timestamp", "")
            try:
                dt = datetime.datetime.fromisoformat(ts)
                time_str = dt.strftime("%I:%M %p")
            except Exception:
                time_str = ts[:16]

            cat   = entry.payload.get("category", "")
            text  = entry.payload.get("text", "")
            ltype = entry.payload.get("log_type", "")
            label = ltype if ltype else cat

            st.markdown(
                f"""
                <div class="entry-card">
                    <div class="entry-time">{time_str}{cat_pill(cat)}</div>
                    <div class="entry-label">{label}</div>
                    <div class="entry-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# 💬  CHAT
# ══════════════════════════════════════════════════════════════════════════════
with chat_tab:
    st.markdown("### Chat to your brain")

    with st.expander("Filter by category (optional — leave blank to search all)"):
        cols = st.columns(3)
        for idx, cat in enumerate(CATEGORIES):
            st.session_state.category_filters[cat] = cols[idx % 3].checkbox(
                cat,
                value=st.session_state.category_filters[cat],
                key=f"filter_{cat}"
            )

    selected_categories = [
        c for c, sel in st.session_state.category_filters.items() if sel
    ]

    query = st.chat_input("Ask anything about your notes...")
    if query:
        with st.spinner("Thinking..."):
            try:
                result = query_notes(question=query,
                                     selected_categories=selected_categories)
                st.session_state.chat_history.append((query, result["answer"]))
            except Exception as e:
                st.error(f"Query error: {str(e)}")

    for q, a in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
