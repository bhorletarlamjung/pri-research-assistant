import os
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------
# Page config
# -----------------------------

st.set_page_config(
    page_title="PRI Research Assistant",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 PRI Research Assistant")

st.write(
    "Ask questions about Paudelian Research Institute’s articles, research themes, "
    "public-benefit mission, and donor information."
)

st.caption(
    "Informational support only. This assistant does not provide legal, tax, medical, "
    "investment, or official government advice."
)


# -----------------------------
# Paths and environment
# -----------------------------

BASE_DIR = Path(__file__).parent
DB_PATH = str(BASE_DIR / "pri_db")

load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)


def get_api_key():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"].strip()
    except Exception:
        pass

    return os.getenv("OPENAI_API_KEY", "").strip()


api_key = get_api_key()

if not api_key:
    st.error("OPENAI_API_KEY is missing. Add it to .env locally or Streamlit Secrets after deployment.")
    st.stop()


# -----------------------------
# Clients
# -----------------------------

try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    st.error("Could not initialize OpenAI client.")
    st.exception(e)
    st.stop()


try:
    chroma = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma.get_or_create_collection(name="pri_website")
except Exception as e:
    st.error("Could not connect to the PRI ChromaDB database.")
    st.exception(e)
    st.stop()


# -----------------------------
# Sidebar
# -----------------------------

try:
    document_count = collection.count()
except Exception:
    document_count = 0

with st.sidebar:
    st.header("PRI Assistant Status")
    st.write("Database path:")
    st.code(DB_PATH)

    st.write("Documents in DB:")
    if document_count > 0:
        st.success(document_count)
    else:
        st.warning("0 documents found")

    st.divider()

    st.write("Suggested questions:")
    st.markdown(
        """
        - What is Paudelian Research Institute?
        - What is Cybernetic Capitalism?
        - What research themes does PRI focus on?
        - What is PRI's public-benefit mission?
        - How can donors support PRI?
        """
    )


# -----------------------------
# Functions
# -----------------------------

def embed(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    except Exception as e:
        st.error("Embedding failed. Please check your OpenAI API key, billing, or internet connection.")
        st.exception(e)
        return None


def retrieve_context(question, n_results=5):
    if document_count == 0:
        return "", []

    question_embedding = embed(question)

    if question_embedding is None:
        return "", []

    try:
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=n_results
        )

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not docs:
            return "", []

        context_parts = []
        sources = []

        for doc, meta in zip(docs, metadatas):
            source = meta.get("source", "Unknown source") if meta else "Unknown source"
            sources.append(source)
            context_parts.append(f"Source: {source}\nContent: {doc}")

        context = "\n\n---\n\n".join(context_parts)

        return context, sorted(set(sources))

    except Exception as e:
        st.error("Could not search the PRI knowledge database.")
        st.exception(e)
        return "", []


def generate_answer(question, context):
    system_prompt = """
You are PRI Research Assistant.

You answer questions using only the provided PRI website context.

Tone:
- professional
- calm
- reader-friendly
- donor-friendly
- careful
- intellectually serious but simple

Rules:
- Do not invent facts.
- If the answer is not in the context, say: "I do not have enough information from the PRI context to answer that confidently."
- Do not give legal, tax, medical, investment, or official government advice.
- For donation questions, say donations may be tax-deductible to the extent permitted by law.
- Do not exaggerate PRI's status.
- Do not claim government, NASA, DARPA, IRS, or university endorsement unless the context explicitly says so.
- Keep answers clear and concise.
- When helpful, use short bullet points.
"""

    user_prompt = f"""
Context:
{context}

Question:
{question}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception as e:
        st.error("Answer generation failed. Please check your OpenAI model access, API key, billing, or network connection.")
        st.exception(e)
        return "I could not generate an answer right now because the language model request failed."


# -----------------------------
# Chat memory
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# -----------------------------
# Chat input
# -----------------------------

question = st.chat_input("Ask PRI Research Assistant...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching PRI knowledge base..."):
            context, sources = retrieve_context(question)

            if not context:
                answer = "I do not have enough information from the PRI context to answer that confidently."
            else:
                answer = generate_answer(question, context)

            st.write(answer)

            if sources:
                with st.expander("Sources used"):
                    for src in sources:
                        st.write(src)

    st.session_state.messages.append({"role": "assistant", "content": answer})


# -----------------------------
# Reset
# -----------------------------

st.divider()

if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()