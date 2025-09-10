import os
import sys
import asyncio
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from chromadb.config import Settings

# --- Fix for sqlite3 on Streamlit Cloud ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # On local Windows, sqlite3 already exists

# --- Fix event loop issue for Streamlit ---
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- Configure Gemini ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("⚠️ Please set your GOOGLE_API_KEY as an environment variable or in Streamlit Cloud secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
gemini = genai.GenerativeModel("gemini-pro")

# --- Step 1: Build or Load Chroma DB ---
DB_DIR = "chroma_db"

if not os.path.exists(DB_DIR):
    st.info("No database found. Building from docs/...")

    # Ensure docs folder exists
    if not os.path.exists("docs"):
        os.makedirs("docs")
        with open("docs/sample.txt", "w") as f:
            f.write("Employees are entitled to 20 days of paid leave per year.")

    loader = DirectoryLoader("docs", glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_DIR,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True, allow_reset=True)
    )
    db.persist()
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True, allow_reset=True)
    )

retriever = db.as_retriever(search_kwargs={"k": 2})

# --- Streamlit UI ---
st.set_page_config(page_title="Company Policy Assistant", page_icon="🏢", layout="wide")
st.title("💬 Company Policy Assistant")
st.write("Your AI assistant for quick answers about company policies.")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/office/80/organization.png", width=80)
    st.markdown("## 🏢 Company Policy Assistant")
    st.markdown("Ask questions about your company's policies (HR, IT, Expenses, etc.)")
    st.markdown("---")
    st.markdown(
        "**Instructions:**\n1. Type a question below\n2. Get instant AI-powered answers\n3. Use for HR, IT, PTO, Expense & Remote work queries"
    )

# --- Chat Interface ---
if "history" not in st.session_state:
    st.session_state.history = []

prompt = st.chat_input("Ask a question about your company policies...")

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})

    with st.spinner("Finding answer..."):
        try:
            # 1. Retrieve relevant docs
            docs = retriever.get_relevant_documents(prompt)
            context = "\n\n".join([doc.page_content for doc in docs])

            # 2. Send to Gemini
            final_prompt = f"""Use the following company policy context to answer the question.
If the answer is not in the context, reply with "I don't know".

Context:
{context}

Question: {prompt}
Answer:"""

            response = gemini.generate_content(final_prompt)
            answer = response.text if response else "I don't know."

            st.session_state.history.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- Display chat history ---
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

