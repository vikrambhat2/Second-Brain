import os
import datetime
import uuid
import logging
from typing import Dict, Optional
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, PayloadSchemaType
import PyPDF2
import docx
from bs4 import BeautifulSoup
import io

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')]
)
logger = logging.getLogger(__name__)

# Predefined categories
CATEGORIES = [
    "Work Projects",
    "Career & Goals",
    "Personal Life",
    "Health & Fitness",
    "Finances",
    "Learning & Courses",
    "Ideas & Inspiration",
    "Travel Plans",
    "Shopping Lists",
    "Tech Notes",
    "Daily Tasks",
    "Random Thoughts"
]

# Load environment variables
load_dotenv()

# Initialize components
try:
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_CLOUD_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    llm = ChatGroq(
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    COLLECTION_NAME = "second-brain-index"
except Exception as e:
    logger.error(f"Initialization error: {e}")
    st.error("Failed to initialize components. Check Qdrant credentials and dependencies.")
    st.stop()

# Initialize Qdrant collection
def init_qdrant():
    try:
        if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="category",
            field_schema=PayloadSchemaType.KEYWORD
        )
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="text",
            field_schema=PayloadSchemaType.TEXT
        )
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="timestamp",
            field_schema=PayloadSchemaType.KEYWORD
        )
    except Exception as e:
        logger.error(f"Qdrant init error: {e}")
        st.error("Qdrant initialization failed.")
        st.stop()

def extract_text(content: bytes, source: str) -> str:
    """Extract text from uploaded file (PDF, DOCX, TXT, HTML)."""
    try:
        content = io.BytesIO(content)
        if source.endswith(".txt"):
            return content.read().decode("utf-8")
        elif source.endswith(".pdf"):
            reader = PyPDF2.PdfReader(content)
            return " ".join(p.extract_text() for p in reader.pages if p.extract_text())
        elif source.endswith(".docx"):
            doc = docx.Document(content)
            return " ".join(p.text for p in doc.paragraphs)
        elif source.endswith(".html"):
            soup = BeautifulSoup(content.read(), "html.parser")
            return soup.get_text()
        else:
            return ""
    except Exception as e:
        logger.error(f"Extraction error for {source}: {e}")
        return ""

def add_note(text: str, category: str = "") -> Dict:
    """Add a note to the Qdrant collection."""
    try:
        note = {
            "id": str(uuid.uuid4()),
            "text": text,
            "category": category,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        vector = embedder.embed_query(note["text"])
        point = PointStruct(
            id=note["id"],
            vector=vector,
            payload={
                "text": note["text"],
                "timestamp": note["timestamp"],
                "category": note["category"],
                "source": category  # Store category as source for consistency
            }
        )
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])
        return {"status": "success", "note_id": note["id"]}
    except Exception as e:
        logger.error(f"Error storing note: {e}")
        return {"status": "error", "message": str(e)}

def query_notes(question: str, selected_categories: Optional[list] = None, top_k: int = 5) -> Dict:
    """Query notes from Qdrant and generate an answer using the LLM."""
    try:
        query_vector = embedder.embed_query(question)
        filters = None
        
        if selected_categories and len(selected_categories) > 0:
            filters = Filter(
                should=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=cat)
                    ) for cat in selected_categories
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
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                f"Today is {today_str}.\n\n"
                "You are a memory assistant. You are only allowed to summarize the notes provided. "
                "Do not make assumptions or add information not explicitly present. "
                "If the question asks about 'today', only use notes that contain today's timestamp. "
                "Respond clearly and concisely."
                "Use the user's previously stored notes to accurately and clearly answer their question. "
                "Point user to the right notes based on the question."
                "Notes are related to  one of the following topics Work, Personal, Health, Finance, Learning, Ideas, Travel, Shopping, Relationships, Tech Notes, TO-DO, Others."   
                "Summarize and synthesize relevant information from the notes provided. "
                "If the notes do not contain enough information, respond honestly with 'Not enough information in memory to answer.'"
            )),
            HumanMessage(content=(
                f"User asked: {question}\n\n"
                f"Relevant notes:\n{context}\n\n"
                f"Answer based only on these notes."
            ))
        ])

        response = llm.invoke(prompt.format())
        return {
            "answer": response.content,
            "results": [{"source": r.payload.get("source", "Unknown"), "timestamp": r.payload.get("timestamp", ""), "text": r.payload.get("text", "")} for r in results]
        }
    except Exception as e:
        logger.error(f"Query error: {e}")
        return {"answer": f"Error while querying notes: {str(e)}", "results": []}

# Streamlit UI
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    password = st.text_input("Enter password to access the app", type="password")
    if password == "vikrambhat":
        st.session_state["authenticated"] = True
        st.rerun()
    elif password:
        st.error("Incorrect password")
    st.stop()

st.set_page_config(page_title="Second Brain", layout="wide")
st.title("\U0001F9E0 Second Brain")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create tabs
store_tab, query_tab = st.tabs(["Store in Brain", "Chat to your brain"])
with store_tab:
    st.header("Store in Brain")

    # -- CATEGORY CHECKBOXES (Separate Session State) --
    st.subheader("Select one category")
    store_cols = st.columns(3)
    store_selected_categories = []

    for idx, cat in enumerate(CATEGORIES):
        key = f"store_cat_{cat}"
        if store_cols[idx % 3].checkbox(cat, key=key):
            store_selected_categories.append(cat)

    # -- TEXT AND FILE INPUT --
    text_input = st.text_area("Paste or type your notes here", height=200)
    uploaded_file = st.file_uploader("Upload a file (optional)", type=["pdf", "docx", "txt", "html"])

    if st.button("Store Memory"):
        with st.spinner("Processing..."):
            try:
                if len(store_selected_categories) != 1:
                    st.warning("Please select exactly one category.")
                    st.stop()

                category = store_selected_categories[0]  # Exactly one

                # Process typed note
                if text_input:
                    result = add_note(text=text_input, category=category)
                    if result["status"] == "success":
                        st.success("Text stored successfully!")
                    else:
                        st.error(f"Error: {result['message']}")

                # Process uploaded file
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

with query_tab:
    st.header("Chat to your brain")

    # Step 1: Initialize checkbox states
    if "category_filters" not in st.session_state:
        st.session_state["category_filters"] = {cat: False for cat in CATEGORIES}

    # Step 2: Render checkboxes with persisted state
    cols = st.columns(3)
    for idx, category in enumerate(CATEGORIES):
        col = cols[idx % 3]
        st.session_state["category_filters"][category] = col.checkbox(
            category,
            value=st.session_state["category_filters"][category],
            key=f"filter_{category}"
        )

    # Step 3: Build the selected categories list
    selected_categories = [
        category for category, selected in st.session_state["category_filters"].items() if selected
    ]
    print("selected_categories",selected_categories)

    query = st.chat_input("Ask a question")
    if query:
        with st.spinner("Retrieving..."):
            try:
                result = query_notes(
                    question=query,
                    selected_categories=selected_categories
                )
                st.session_state.chat_history.append((query, result["answer"]))
            except Exception as e:
                st.error(f"Query error: {str(e)}")

    # Chat-style display
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

init_qdrant()
