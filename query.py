from langchain.vectorstores import Chroma
from llm import get_gemini_response

# Load vector DB
db = Chroma(persist_directory="db", embedding_function=None)
retriever = db.as_retriever()

def answer_question(question):
    # Retrieve relevant chunks
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"Answer the question using the following company policies:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    answer = get_gemini_response(prompt)
    return answer
