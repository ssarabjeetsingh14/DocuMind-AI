```python
from dotenv import load_dotenv

from langchain_mistralai import (
    MistralAIEmbeddings,
    ChatMistralAI
)

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ----------------- LOAD ENV -----------------

load_dotenv()


# ----------------- EMBEDDING MODEL -----------------

embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)


# ----------------- LOAD CHROMA DATABASE -----------------

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)


# ----------------- RETRIEVER -----------------

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)


# ----------------- LLM -----------------

llm = ChatMistralAI(
    model="ministral-8b-latest",
    temperature=0.3
)


# ----------------- PROMPT -----------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say:

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


# ----------------- RAG SYSTEM READY -----------------

print("RAG system created successfully!")
print("Press 0 to exit.")


# ----------------- CHAT LOOP -----------------

while True:

    query = input("\nYou: ")

    if query == "0":
        print("Goodbye!")
        break


    # ----------------- RETRIEVE -----------------

    try:

        docs = retriever.invoke(query)

    except Exception as e:

        print("\n❌ Error while retrieving documents:")
        print(e)
        break


    # ----------------- CREATE CONTEXT -----------------

    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
            if isinstance(doc.page_content, str)
        ]
    )


    # ----------------- CREATE PROMPT -----------------

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )


    # ----------------- CALL LLM -----------------

    try:

        response = llm.invoke(final_prompt)

    except Exception as e:

        print("\n❌ Error while calling Mistral:")
        print(e)
        continue


    # ----------------- RESPONSE -----------------

    print(f"\nAI: {response.content}")
```
