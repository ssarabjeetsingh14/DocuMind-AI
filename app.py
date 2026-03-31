import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.memory import ConversationBufferMemory

# ----------------- SETUP -----------------
load_dotenv()
st.set_page_config(page_title="DocuMind AI", layout="wide")

st.title("📚 DocuMind AI – Chat with Your PDF")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
    k = st.slider("Top K Results", 1, 10, 4)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.3)

# ----------------- SESSION STATE -----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# ----------------- FILE UPLOAD -----------------
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ PDF uploaded!")

    with st.spinner("🔄 Processing PDF..."):
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        embedding_model = MistralAIEmbeddings()

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

        st.session_state.vectorstore = vectorstore

    st.success("✅ Database ready!")

# ----------------- CHAT UI -----------------
if st.session_state.vectorstore:

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-2506",
        temperature=temperature
    )

    # Prompt with memory
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a helpful AI assistant.
Use ONLY the provided context to answer.
If not found, say: "I could not find the answer in the document."
"""),
            ("placeholder", "{history}"),
            ("human", """Context:
{context}

Question:
{question}
""")
        ]
    )

    memory = st.session_state.memory

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if user_input := st.chat_input("Ask something about your PDF..."):

        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve context
        docs = retriever.invoke(user_input)
        context = "\n\n".join([doc.page_content for doc in docs])

        history = memory.load_memory_variables({})["history"]

        final_prompt = prompt.invoke({
            "context": context,
            "question": user_input,
            "history": history
        })

        response = llm.invoke(final_prompt)

        # Save memory
        memory.save_context(
            {"input": user_input},
            {"output": response.content}
        )

        # Show assistant response
        with st.chat_message("assistant"):
            st.markdown(response.content)

            # -------- SOURCES --------
            with st.expander("📌 Sources"):
                for i, doc in enumerate(docs):
                    st.write(f"**Source {i+1}:**")
                    st.write(doc.page_content[:300] + "...")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.content
        })

else:
    st.info("👆 Upload a PDF to start chatting.")