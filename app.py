import os
import sys
import asyncio
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from chromadb.config import Settings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv

# --- Fix for sqlite3 on Streamlit Cloud ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # on local Windows, sqlite3 already exists

# --- Fix for chromadb/sqlite3 on Streamlit Cloud ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- Fix event loop issue for Streamlit ---
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- Step 1: Build or Load Database ---
DB_DIR = "chroma_db"

if not os.path.exists(DB_DIR):
    st.info("No database found. Building from docs/ ...")

    # Ensure docs folder exists
    if not os.path.exists("docs"):
        os.makedirs("docs")
        with open("docs/sample.txt", "w") as f:
            f.write("This is a sample company policy document.")

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

# --- Step 2: Setup LLM ---
# ⚠️ Choose a model: small for faster load, big if you want better answers
# Small model (good for free Streamlit Cloud)
model_id = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=pipe)

# --- Step 3: Setup Retriever & QA ---
retriever = db.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- Step 4: Streamlit UI ---
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
        "**Instructions:**\\n1. Type a question below\\n2. Get instant AI-powered answers\\n3. Use for HR, IT, PTO, Expense & Remote work queries"
    )

# --- Chat Interface ---
if "history" not in st.session_state:
    st.session_state.history = []
<<<<<<< HEAD
=======

prompt = st.chat_input("Ask a question about your company policies...")

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})

    with st.spinner("Finding answer..."):
        try:
            answer = qa.run(prompt)
            st.session_state.history.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Display chat history
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
>>>>>>> 7548062 (Initial commit for Streamlit deployment)
