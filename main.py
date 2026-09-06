```python
from dotenv import load_dotenv

from langchain_mistralai import (
    MistralAIEmbeddings,
    ChatMistralAI
)

# Use the standalone Chroma integration
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)


# =========================================================
# LOAD CHROMA DATABASE
# =========================================================

vectorstore = Chroma(
    collection_name="documind_collection",
    persist_directory="chroma_db",
    embedding_function=embedding_model
)


# =========================================================
# RETRIEVER
# =========================================================

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)


# =========================================================
# MISTRAL LLM
# =========================================================

llm = ChatMistralAI(
    model="ministral-8b-latest",
    temperature=0.3
)


# =========================================================
# PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context, say:

"I could not find the answer in the document."

Do not make up information.
"""
        ),

        (
            "human",
            """Context:

{context}

Question:

{question}
"""
        )
    ]
)


# =========================================================
# RAG SYSTEM READY
# =========================================================

print("=" * 60)
print("📚 DocuMind AI")
print("=" * 60)
print("✅ RAG system created successfully!")
print("Type 0 to exit.")
print("=" * 60)


# =========================================================
# CHAT LOOP
# =========================================================

while True:

    query = input("\nYou: ").strip()

    # Exit
    if query == "0":
        print("\nGoodbye! 👋")
        break

    # Empty input
    if not query:
        print("Please enter a question.")
        continue


    # =====================================================
    # RETRIEVE DOCUMENTS
    # =====================================================

    try:

        docs = retriever.invoke(query)

    except Exception as e:

        print("\n❌ Error while retrieving documents:")
        print(type(e).__name__)
        print(e)

        continue


    # =====================================================
    # CHECK RETRIEVAL
    # =====================================================

    if not docs:

        print(
            '\nAI: I could not find the answer in the document.'
        )

        continue


    # =====================================================
    # CREATE CONTEXT
    # =====================================================

    valid_docs = [
        doc
        for doc in docs
        if isinstance(doc.page_content, str)
        and doc.page_content.strip()
    ]

    context = "\n\n".join(
        doc.page_content
        for doc in valid_docs
    )


    # =====================================================
    # CHECK CONTEXT
    # =====================================================

    if not context.strip():

        print(
            '\nAI: I could not find the answer in the document.'
        )

        continue


    # =====================================================
    # CREATE FINAL PROMPT
    # =====================================================

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )


    # =====================================================
    # CALL MISTRAL
    # =====================================================

    try:

        response = llm.invoke(final_prompt)

    except Exception as e:

        print("\n❌ Error while calling Mistral:")
        print(type(e).__name__)
        print(e)

        continue


    # =====================================================
    # DISPLAY RESPONSE
    # =====================================================

    print(f"\nAI: {response.content}")


    # =====================================================
    # DISPLAY SOURCES
    # =====================================================

    print("\n📌 Sources:")

    for i, doc in enumerate(valid_docs, start=1):

        page = doc.metadata.get("page", "Unknown")

        print(
            f"  Source {i} | Page: {page + 1 if isinstance(page, int) else page}"
        )
```
