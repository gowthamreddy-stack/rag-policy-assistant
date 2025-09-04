import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from chromadb.config import Settings

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

# --- Load documents ---
if not os.path.exists("docs"):
    os.makedirs("docs")
    # Add at least one sample document for deployment
    with open("docs/sample.txt", "w") as f:
        f.write("This is a sample company policy document.")

loader = DirectoryLoader("docs", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

# --- Split text into chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# --- Setup embeddings ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# --- Create Chroma DB ---
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db",
    client_settings=Settings(
        anonymized_telemetry=False,
        is_persistent=True,
        allow_reset=True
    )
)

db.persist()
print("✅ Chroma DB created and persisted in ./chroma_db")