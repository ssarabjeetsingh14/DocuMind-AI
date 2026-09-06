import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_mistralai import (
    MistralAIEmbeddings,
    ChatMistralAI
)

# Standalone Chroma integration
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.memory import ConversationBufferMemory


# =========================================================
# SETUP
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 DocuMind AI – Chat with Your PDF")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    chunk_size = st.slider(
        "Chunk Size",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100
    )

    k = st.slider(
        "Top K Results",
        min_value=1,
        max_value=10,
        value=4
    )

    temperature = st.slider(
        "LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )


# =========================================================
# SESSION STATE
# =========================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True
    )

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "uploaded_file_size" not in st.session_state:
    st.session_state.uploaded_file_size = None

if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = None


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file:

    current_file_name = uploaded_file.name
    current_file_size = uploaded_file.size

    file_changed = (
        st.session_state.uploaded_file_name
        != current_file_name
        or
        st.session_state.uploaded_file_size
        != current_file_size
        or
        st.session_state.chunk_size
        != chunk_size
    )


    # =====================================================
    # PROCESS PDF ONLY WHEN NECESSARY
    # =====================================================

    if file_changed:

        with st.spinner(
            "🔄 Processing PDF and creating embeddings..."
        ):

            try:

                # -----------------------------------------
                # SAVE TEMP PDF
                # -----------------------------------------

                pdf_path = "temp.pdf"

                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())


                # -----------------------------------------
                # LOAD PDF
                # -----------------------------------------

                loader = PyPDFLoader(pdf_path)

                docs = loader.load()


                # -----------------------------------------
                # REMOVE EMPTY PAGES
                # -----------------------------------------

                docs = [
                    doc
                    for doc in docs
                    if isinstance(doc.page_content, str)
                    and doc.page_content.strip()
                ]


                if not docs:

                    st.error(
                        "❌ No readable text was found in this PDF."
                    )

                    st.stop()


                # -----------------------------------------
                # TEXT SPLITTER
                # -----------------------------------------

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(docs)


                # -----------------------------------------
                # REMOVE EMPTY CHUNKS
                # -----------------------------------------

                chunks = [
                    chunk
                    for chunk in chunks
                    if isinstance(chunk.page_content, str)
                    and chunk.page_content.strip()
                ]


                if not chunks:

                    st.error(
                        "❌ No text chunks were created."
                    )

                    st.stop()


                # -----------------------------------------
                # EMBEDDING MODEL
                # -----------------------------------------

                embedding_model = MistralAIEmbeddings(
                    model="mistral-embed"
                )


                # -----------------------------------------
                # CREATE IN-MEMORY CHROMA DATABASE
                # -----------------------------------------

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    collection_name="documind_collection"
                )


                # -----------------------------------------
                # SAVE TO SESSION STATE
                # -----------------------------------------

                st.session_state.vectorstore = vectorstore

                st.session_state.uploaded_file_name = (
                    current_file_name
                )

                st.session_state.uploaded_file_size = (
                    current_file_size
                )

                st.session_state.chunk_size = chunk_size


                # -----------------------------------------
                # RESET CHAT
                # -----------------------------------------

                st.session_state.messages = []

                st.session_state.memory = (
                    ConversationBufferMemory(
                        return_messages=True
                    )
                )


                st.success(
                    f"✅ PDF processed successfully! "
                    f"{len(chunks)} chunks created."
                )


            except Exception as e:

                st.error(
                    "❌ Error while processing the PDF."
                )

                st.exception(e)

                st.stop()


# =========================================================
# CHAT
# =========================================================

if st.session_state.vectorstore:

    # =====================================================
    # RETRIEVER
    # =====================================================

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )


    # =====================================================
    # MISTRAL LLM
    # =====================================================

    llm = ChatMistralAI(
        model="ministral-8b-latest",
        temperature=temperature
    )


    # =====================================================
    # PROMPT
    # =====================================================

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer cannot be found in the context, say:

"I could not find the answer in the document."

Do not make up information.
"""
            ),

            (
                "placeholder",
                "{history}"
            ),

            (
                "human",
                """Context:

{context}

Question:

{question}"""
            )
        ]
    )


    # =====================================================
    # MEMORY
    # =====================================================

    memory = st.session_state.memory


    # =====================================================
    # DISPLAY PREVIOUS MESSAGES
    # =====================================================

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])


    # =====================================================
    # USER INPUT
    # =====================================================

    user_input = st.chat_input(
        "Ask something about your PDF..."
    )


    if user_input:

        # -----------------------------------------------
        # DISPLAY USER MESSAGE
        # -----------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        with st.chat_message("user"):

            st.markdown(user_input)


        # -----------------------------------------------
        # RETRIEVE DOCUMENTS
        # -----------------------------------------------

        try:

            docs = retriever.invoke(user_input)

        except Exception as e:

            st.error(
                "❌ Error while searching the vector database."
            )

            st.exception(e)

            st.stop()


        # -----------------------------------------------
        # FILTER VALID DOCUMENTS
        # -----------------------------------------------

        valid_docs = [
            doc
            for doc in docs
            if isinstance(doc.page_content, str)
            and doc.page_content.strip()
        ]


        # -----------------------------------------------
        # CREATE CONTEXT
        # -----------------------------------------------

        context = "\n\n".join(
            doc.page_content
            for doc in valid_docs
        )


        # -----------------------------------------------
        # GET CONVERSATION HISTORY
        # -----------------------------------------------

        history = memory.load_memory_variables({})[
            "history"
        ]


        # -----------------------------------------------
        # CREATE FINAL PROMPT
        # -----------------------------------------------

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": user_input,
                "history": history
            }
        )


        # -----------------------------------------------
        # CALL MISTRAL
        # -----------------------------------------------

        try:

            response = llm.invoke(final_prompt)

        except Exception as e:

            st.error(
                "❌ Error while calling Mistral AI."
            )

            st.exception(e)

            st.stop()


        # -----------------------------------------------
        # SAVE MEMORY
        # -----------------------------------------------

        memory.save_context(
            {
                "input": user_input
            },
            {
                "output": response.content
            }
        )


        # -----------------------------------------------
        # DISPLAY AI RESPONSE
        # -----------------------------------------------

        with st.chat_message("assistant"):

            st.markdown(response.content)


            # -------------------------------------------
            # SOURCES
            # -------------------------------------------

            with st.expander("📌 Sources"):

                if valid_docs:

                    for i, doc in enumerate(
                        valid_docs,
                        start=1
                    ):

                        page = doc.metadata.get(
                            "page",
                            None
                        )

                        if isinstance(page, int):

                            source_title = (
                                f"Source {i} — "
                                f"Page {page + 1}"
                            )

                        else:

                            source_title = (
                                f"Source {i}"
                            )


                        st.write(
                            f"**{source_title}**"
                        )

                        st.write(
                            doc.page_content[:500]
                            + "..."
                        )

                else:

                    st.write(
                        "No sources were retrieved."
                    )


        # -----------------------------------------------
        # SAVE AI MESSAGE
        # -----------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.content
            }
        )


# =========================================================
# NO PDF
# =========================================================

else:

    st.info(
        "👆 Upload a PDF to start chatting."
    )
