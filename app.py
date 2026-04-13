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

# ── Page config must be the first Streamlit call ─────────────────────────────
st.set_page_config(
    page_title="Second Brain",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')]
)
logger = logging.getLogger(__name__)

# ── Categories ────────────────────────────────────────────────────────────────
CATEGORIES = [
    "Journal",
    "Personal Life",
    "Family",
    "Health & Fitness",
    "Finances",
    "Work Projects",
    "Career & Goals",
    "Ideas & Inspiration",
    "Travel Plans",
    "Shopping Lists",
    "Daily Tasks",
    "Random Thoughts",
]

# Quick log types and their default categories
QUICK_LOG_TYPES = [
    "💭 Free Thought",
    "😊 Mood Check-in",
    "😴 Sleep Log",
    "🏃 Exercise",
    "💰 Expense",
    "🍽️ Meal",
    "💼 Work Session",
    "💡 Idea",
    "👥 Social Interaction",
]

QUICK_LOG_CATEGORY = {
    "💭 Free Thought":       "Random Thoughts",
    "😊 Mood Check-in":      "Journal",
    "😴 Sleep Log":          "Health & Fitness",
    "🏃 Exercise":           "Health & Fitness",
    "💰 Expense":            "Finances",
    "🍽️ Meal":              "Health & Fitness",
    "💼 Work Session":       "Work Projects",
    "💡 Idea":               "Ideas & Inspiration",
    "👥 Social Interaction": "Personal Life",
}

# ── Load env & init clients ───────────────────────────────────────────────────
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

# ── Qdrant collection setup ───────────────────────────────────────────────────
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
    st.title("🧠 Second Brain")
    password = st.text_input("Password", type="password", placeholder="Enter password...")
    if st.button("Unlock", use_container_width=True):
        if password == os.getenv("APP_PASSWORD", ""):
            st.session_state["authenticated"] = True
            st.rerun()
        elif password:
            st.error("Incorrect password")
    st.stop()

# ── Core functions ────────────────────────────────────────────────────────────

def add_note(text: str, category: str, extra_payload: dict = None) -> Dict:
    """Store a note in Qdrant with optional structured metadata."""
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
        point = PointStruct(id=note_id, vector=vector, payload=payload)
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])
        return {"status": "success", "note_id": note_id}
    except Exception as e:
        logger.error(f"Error storing note: {e}")
        return {"status": "error", "message": str(e)}


def get_today_entries() -> list:
    """Fetch all entries logged today (UTC), sorted newest first."""
    try:
        today_start = datetime.datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        today_unix = int(today_start.timestamp())
        results = []
        offset = None
        while True:
            batch, next_offset = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="timestamp_unix",
                            range=Range(gte=today_unix)
                        )
                    ]
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
    """Extract text from uploaded file (PDF, DOCX, TXT, HTML)."""
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
            soup = BeautifulSoup(buf.read(), "html.parser")
            return soup.get_text()
        return ""
    except Exception as e:
        logger.error(f"Extraction error for {source}: {e}")
        return ""


def query_notes(question: str, selected_categories: Optional[list] = None, top_k: int = 5) -> Dict:
    """Semantic search over notes and generate an LLM answer."""
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
                "You are a personal memory assistant. Answer only based on the notes provided — "
                "do not use assumptions or external knowledge. Be clear and concise. "
                "If there is not enough information, say: 'Not enough information in memory to answer.'"
            )),
            HumanMessage(content=(
                f"Question: {question}\n\nRelevant notes:\n{context}\n\nAnswer based only on these notes."
            ))
        ]
        response = llm.invoke(messages)
        return {
            "answer": response.content,
            "results": [
                {
                    "category": r.payload.get("category", ""),
                    "timestamp": r.payload.get("timestamp", ""),
                    "text": r.payload.get("text", "")
                }
                for r in results
            ]
        }
    except Exception as e:
        logger.error(f"Query error: {e}")
        return {"answer": f"Error while querying: {str(e)}", "results": []}


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🧠 Second Brain")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

quick_tab, store_tab, feed_tab, chat_tab = st.tabs([
    "⚡ Quick Log", "📥 Store", "📅 Today", "💬 Chat"
])


# ── QUICK LOG ─────────────────────────────────────────────────────────────────
with quick_tab:
    log_type = st.selectbox(
        "What are you logging?",
        QUICK_LOG_TYPES,
        label_visibility="collapsed"
    )

    category = QUICK_LOG_CATEGORY[log_type]
    note_text = ""
    extra_payload = {"log_type": log_type}

    st.divider()

    if log_type == "💭 Free Thought":
        note_text = st.text_area(
            "What's on your mind?",
            height=150,
            placeholder="Just type...",
            label_visibility="collapsed"
        )
        category = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(category))

    elif log_type == "😊 Mood Check-in":
        col1, col2 = st.columns(2)
        mood = col1.slider("Mood", 1, 10, 7)
        energy = col2.slider("Energy", 1, 10, 7)
        mood_emojis = {1:"😭",2:"😢",3:"😟",4:"😕",5:"😐",6:"🙂",7:"😊",8:"😄",9:"😁",10:"🤩"}
        st.caption(f"Mood: {mood_emojis.get(mood,'')}  ·  Energy: {'⚡' * min(energy, 5)}")
        note = st.text_area("How are you feeling? (optional)", height=100,
                            placeholder="What's going on?")
        note_text = f"Mood check-in — mood {mood}/10, energy {energy}/10. {note}".strip(" .")
        extra_payload.update({"mood_score": mood, "energy_score": energy})

    elif log_type == "😴 Sleep Log":
        col1, col2 = st.columns(2)
        hours = col1.number_input("Hours slept", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        quality = col2.slider("Quality (1–5)", 1, 5, 3)
        quality_labels = {1:"terrible", 2:"poor", 3:"okay", 4:"good", 5:"excellent"}
        note = st.text_area("Notes (optional)", height=80,
                            placeholder="Dreams, interruptions, how you woke up...")
        note_text = f"Sleep: {hours}h, quality {quality}/5 ({quality_labels[quality]}). {note}".strip(" .")
        extra_payload.update({"sleep_hours": hours, "sleep_quality": quality})

    elif log_type == "🏃 Exercise":
        activity = st.text_input("Activity", placeholder="Running, gym, yoga, walk, swim...")
        col1, col2 = st.columns(2)
        duration = col1.number_input("Duration (mins)", min_value=0, max_value=480, value=30, step=5)
        intensity = col2.selectbox("Intensity", ["Low", "Moderate", "High", "Max"])
        note = st.text_area("Notes (optional)", height=80,
                            placeholder="What did you do? How did it feel?")
        note_text = f"Exercise: {activity}, {duration} mins, {intensity.lower()} intensity. {note}".strip(" .")
        extra_payload.update({"activity": activity, "duration_mins": duration, "intensity": intensity})

    elif log_type == "💰 Expense":
        col1, col2 = st.columns(2)
        amount = col1.number_input("Amount ($)", min_value=0.0, step=0.01, format="%.2f")
        payment = col2.selectbox("Via", ["Cash", "Credit Card", "Debit Card", "UPI", "Other"])
        description = st.text_input("What for?", placeholder="Coffee, groceries, subscription...")
        note = st.text_input("Notes (optional)")
        note_text = f"Expense: ${amount:.2f} via {payment} — {description}. {note}".strip(" .")
        extra_payload.update({"amount": amount, "payment_method": payment})

    elif log_type == "🍽️ Meal":
        meal_type = st.selectbox("Meal", ["Breakfast", "Lunch", "Dinner", "Snack"])
        food = st.text_area("What did you eat?", height=80,
                            placeholder="Be as detailed as you like...")
        col1, col2 = st.columns(2)
        hunger_before = col1.slider("Hunger before (1–5)", 1, 5, 3)
        satisfaction = col2.slider("Satisfaction after (1–5)", 1, 5, 4)
        note = st.text_input("Notes (optional)", placeholder="How did it taste? Where were you?")
        note_text = (
            f"{meal_type}: {food}. "
            f"Hunger before {hunger_before}/5, satisfaction {satisfaction}/5. {note}"
        ).strip(" .")
        extra_payload.update({"meal_type": meal_type, "hunger_before": hunger_before,
                               "satisfaction": satisfaction})

    elif log_type == "💼 Work Session":
        focus = st.text_input("Focus area / project",
                              placeholder="What were you working on?")
        duration = st.number_input("Duration (mins)", min_value=0, max_value=480,
                                   value=60, step=15)
        accomplishment = st.text_area("What did you accomplish?", height=80,
                                      placeholder="Key outputs, decisions, progress...")
        blockers = st.text_input("Blockers (optional)",
                                 placeholder="Anything that slowed you down?")
        note_text = (
            f"Work session on '{focus}', {duration} mins. "
            f"Accomplished: {accomplishment}."
            f"{(' Blockers: ' + blockers) if blockers else ''}"
        ).strip()
        extra_payload.update({"focus_area": focus, "duration_mins": duration})

    elif log_type == "💡 Idea":
        title = st.text_input("Idea title", placeholder="Give it a name...")
        description = st.text_area("Describe the idea", height=120,
                                   placeholder="What is it? Why does it matter?")
        context = st.text_input("Context / what triggered it (optional)")
        note_text = (
            f"Idea — {title}: {description}."
            f"{(' Context: ' + context) if context else ''}"
        ).strip()

    elif log_type == "👥 Social Interaction":
        person = st.text_input("Who?",
                               placeholder="Name or relationship (friend, colleague...)")
        what = st.text_area("What happened?", height=80,
                            placeholder="Describe the interaction...")
        feeling = st.selectbox("How did it make you feel?",
                               ["", "Energised", "Happy", "Neutral",
                                "Uncomfortable", "Drained", "Inspired"])
        note = st.text_input("Notes (optional)")
        note_text = (
            f"Interaction with {person}: {what}."
            f"{(' Feeling: ' + feeling) if feeling else ''}"
            f"{(' ' + note) if note else ''}"
        ).strip()
        extra_payload.update({"person": person, "feeling": feeling})

    st.divider()
    if st.button("⚡ Log It", use_container_width=True, type="primary"):
        if note_text.strip():
            with st.spinner("Saving..."):
                result = add_note(text=note_text, category=category, extra_payload=extra_payload)
            if result["status"] == "success":
                st.success("Logged!")
                st.balloons()
            else:
                st.error(f"Error: {result['message']}")
        else:
            st.warning("Nothing to log — fill in the fields first.")


# ── STORE ─────────────────────────────────────────────────────────────────────
with store_tab:
    st.header("Store in Brain")

    with st.expander("Select a Category"):
        st.caption("Select exactly one category")
        store_cols = st.columns(3)
        store_selected_categories = []
        for idx, cat in enumerate(CATEGORIES):
            if store_cols[idx % 3].checkbox(cat, key=f"store_cat_{cat}"):
                store_selected_categories.append(cat)

    text_input = st.text_area("Paste or type your notes here", height=200)

    with st.expander("Or upload a file"):
        uploaded_file = st.file_uploader(
            "Upload a file (optional)", type=["pdf", "docx", "txt", "html"]
        )

    if st.button("Store Memory", use_container_width=True):
        with st.spinner("Processing..."):
            try:
                if len(store_selected_categories) != 1:
                    st.warning("Please select exactly one category.")
                    st.stop()
                category = store_selected_categories[0]

                if text_input:
                    result = add_note(text=text_input, category=category)
                    if result["status"] == "success":
                        st.success("Text stored successfully!")
                    else:
                        st.error(f"Error: {result['message']}")

                if uploaded_file:
                    content = uploaded_file.read()
                    text = extract_text(content, uploaded_file.name)
                    if text:
                        result = add_note(text=text, category=category)
                        if result["status"] == "success":
                            st.success(f"Stored {uploaded_file.name}!")
                        else:
                            st.error(f"Error storing {uploaded_file.name}: {result['message']}")
                    else:
                        st.error(f"Failed to extract text from {uploaded_file.name}.")
            except Exception as e:
                st.error(f"Error: {str(e)}")


# ── TODAY'S FEED ──────────────────────────────────────────────────────────────
with feed_tab:
    today_label = datetime.datetime.utcnow().strftime("%A, %B %d")
    st.header(f"Today — {today_label}")

    col_refresh, _ = st.columns([1, 5])
    if col_refresh.button("🔄 Refresh"):
        st.rerun()

    with st.spinner("Loading today's entries..."):
        entries = get_today_entries()

    if not entries:
        st.info("No entries yet today. Tap **⚡ Quick Log** to start logging.")
    else:
        # Summary bar
        cat_counts: dict = {}
        for e in entries:
            cat = e.payload.get("category", "Uncategorized")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        num_cols = min(len(cat_counts), 4)
        cols = st.columns(num_cols)
        for i, (cat, count) in enumerate(cat_counts.items()):
            cols[i % num_cols].metric(cat, count)

        st.divider()

        # Timeline
        for entry in entries:
            ts = entry.payload.get("timestamp", "")
            try:
                dt = datetime.datetime.fromisoformat(ts)
                time_str = dt.strftime("%I:%M %p")
            except Exception:
                time_str = ts[:16]

            cat      = entry.payload.get("category", "")
            text     = entry.payload.get("text", "")
            ltype    = entry.payload.get("log_type", "")
            label    = ltype if ltype else cat

            col_time, col_body = st.columns([1, 5])
            col_time.caption(time_str)
            col_body.markdown(f"**{label}**")
            col_body.write(text)
            st.divider()


# ── CHAT ──────────────────────────────────────────────────────────────────────
with chat_tab:
    st.header("Chat to your brain")

    if "category_filters" not in st.session_state:
        st.session_state["category_filters"] = {cat: False for cat in CATEGORIES}

    with st.expander("Filter by category (optional)"):
        cols = st.columns(3)
        for idx, cat in enumerate(CATEGORIES):
            st.session_state["category_filters"][cat] = cols[idx % 3].checkbox(
                cat,
                value=st.session_state["category_filters"][cat],
                key=f"filter_{cat}"
            )

    selected_categories = [
        cat for cat, sel in st.session_state["category_filters"].items() if sel
    ]

    query = st.chat_input("Ask anything about your notes...")
    if query:
        with st.spinner("Retrieving..."):
            try:
                result = query_notes(question=query, selected_categories=selected_categories)
                st.session_state.chat_history.append((query, result["answer"]))
            except Exception as e:
                st.error(f"Query error: {str(e)}")

    for q, a in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
