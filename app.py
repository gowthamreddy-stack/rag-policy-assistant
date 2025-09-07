import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from chromadb.config import Settings
import asyncio

# --- Fix for chromadb/sqlite3 on Streamlit Cloud ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- Fix event loop issue for Streamlit ---
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- Load environment variables ---
try:
    # Use st.secrets for Streamlit deployment
    GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    # Fallback for local development
    from dotenv import load_dotenv
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Add it in .env (local) or secrets.toml (Streamlit).")

# --- Setup embeddings ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# --- Load persisted Chroma DB ---
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    client_settings=Settings(
        anonymized_telemetry=False,
        is_persistent=True,
        allow_reset=True
    )
)

# --- Setup LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

# --- Setup retriever & QA ---
retriever = db.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- Streamlit UI ---
st.set_page_config(page_title="Company Policy Assistant", page_icon="🏢", layout="wide")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/office/80/organization.png", width=80)
    st.markdown("## 🏢 Company Policy Assistant")
    st.markdown("Ask questions about your company's policies (HR, IT, Expenses, etc.)")
    st.markdown("---")
    st.markdown(
        "**Instructions:**\\n1. Type a question below\\n2. Get instant AI-powered answers\\n3. Use for HR, IT, PTO, Expense & Remote work queries"
    )

# Title
st.title("💬 Company Policy Assistant")
st.write("Your AI assistant for quick answers about company policies.")

# --- Chat Interface ---
if "history" not in st.session_state:
    st.session_state.history = []
