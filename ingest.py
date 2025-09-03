import os
import glob
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# --- Load env vars ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file")

# --- Load markdown files ---
documents = []
for file in glob.glob("company_policies/*.md"):
    loader = TextLoader(file, encoding="utf-8")
    documents.extend(loader.load())

if not documents:
    raise ValueError("❌ No documents found in company_policies folder")

# --- Split into chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# --- Setup embeddings ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# --- Create Chroma DB (persist locally) ---
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,   # ✅ use "embedding", not embedding_function
    persist_directory="chroma_db"
)

db.persist()
print(f"✅ Ingested {len(chunks)} chunks into Chroma DB")






    
