import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import asyncio


try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Add it to Streamlit Secrets or your environment variables.")


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)
db = Chroma.from_documents([], embedding_function=embeddings)


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)


retriever = db.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


st.set_page_config(page_title="Company Policy Assistant", page_icon="🏢", layout="wide")


with st.sidebar:
    st.image("https://img.icons8.com/office/80/organization.png", width=80)  
    st.markdown("## 🏢 Company Policy Assistant")
    st.markdown("Ask questions about your company's policies (HR, IT, Expenses, etc.)")
    st.markdown("---")
    st.markdown("**Instructions:**\n1. Type a question below\n2. Get instant AI-powered answers\n3. Use for HR, IT, PTO, Expense & Remote work queries")


st.title("💬 Company Policy Assistant")
st.write("Your AI assistant for quick answers about company policies.")


if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask a question about company policies...")

if query:
    answer = qa.run(query)
    st.session_state.history.append({"question": query, "answer": answer})


for chat in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])


