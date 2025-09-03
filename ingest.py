import os
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Load environment variables at the top
load_dotenv()

def ingest_policies():
    folder = "company_policies"
    persist_dir = "chroma_db"

    # ✅ Use API key from .env
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    # Load documents
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".md"):
            loader = TextLoader(os.path.join(folder, file))
            docs.extend(loader.load())

    if not docs:
        raise ValueError("❌ No documents found in company_policies folder")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # Create Chroma DB (overwrite if needed)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)

    print(f"✅ Ingested {len(chunks)} chunks into Chroma")

if __name__ == "__main__":
    ingest_policies()




    
