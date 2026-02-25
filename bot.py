"""
bot.py — Retrieval + Llama 3.3 70B via Groq API (no HuggingFace needed).

Groq hosts Llama 3.3 70B for free with an API key from https://console.groq.com
Groq is extremely fast (low-latency inference) and has a generous free tier.

Prompting Styles:
  1. Factual      — direct, cite-based answer       (temp=0.1)
  2. Analytical   — deeper interpretation            (temp=0.3)
  3. Summary      — concise bullet-point format      (temp=0.1)
  4. Critical     — gaps, caveats, what's missing    (temp=0.4)
"""

import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from ingest import get_embeddings

load_dotenv()

FAISS_INDEX_PATH = "faiss_index"
_vectorstore = None

# ── Load FAISS vectorstore ─────────────────────────────────────────────────────
def load_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = get_embeddings()
        _vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("[BOT] FAISS vectorstore loaded.")
    return _vectorstore


def get_chain_status():
    try:
        vs = load_vectorstore()
        return {"ready": True, "vector_count": vs.index.ntotal}
    except Exception as e:
        return {"ready": False, "error": str(e)}


# ── Retrieve top-k relevant chunks ────────────────────────────────────────────
def retrieve_chunks(question: str, k: int = 5):
    vs = load_vectorstore()
    return vs.similarity_search(question, k=k)


def format_context(docs) -> str:
    ctx = ""
    for i, doc in enumerate(docs):
        page     = doc.metadata.get("page", "?")
        strategy = doc.metadata.get("chunking", "N/A")
        ctx += f"\n[Chunk {i+1} | Page {page} | Strategy: {strategy}]\n{doc.page_content}\n"
    return ctx.strip()


# ── Groq API call with Llama 3.3 70B ─────────────────────────────────────────
def call_groq(system_prompt: str, user_message: str,
              temperature: float = 0.2, max_tokens: int = 600) -> str:
    """
    Calls Groq's hosted Llama-3.3-70B-Versatile model.
    Get your free API key at: https://console.groq.com
    No HuggingFace account needed.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com and add it to your .env file."
        )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",      # Llama 3.3 70B hosted on Groq
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ── 4 Prompting Strategies ─────────────────────────────────────────────────────
PROMPTS = {
    "factual": {
        "label":       "📋 Factual Answer",
        "description": "Direct, precise answer citing the document",
        "temperature": 0.1,
        "system": (
            "You are a precise document analyst. "
            "Answer the question ONLY using the provided context. "
            "Be direct, specific, and cite page numbers when possible. "
            "If information is not found, state: 'Not mentioned in the document.' "
            "Do not speculate beyond the text."
        ),
    },
    "analytical": {
        "label":       "🔍 Analytical Interpretation",
        "description": "Deeper analysis and implications from the document",
        "temperature": 0.3,
        "system": (
            "You are an expert document interpreter and business analyst. "
            "Using the provided context, go beyond the surface answer — "
            "interpret the implications, connect related points, and explain "
            "what this means in a broader context. "
            "Base your answer on the document but provide deeper insight."
        ),
    },
    "summary": {
        "label":       "📌 Key Points Summary",
        "description": "Concise bullet-point format of the most important info",
        "temperature": 0.1,
        "system": (
            "You are a professional summarizer. "
            "Answer in bullet-point format using ONLY the context provided. "
            "Limit to 5 bullets maximum. Start each with '•'. "
            "Be concise — 1-2 sentences per bullet. "
            "If information is not in the context, say so briefly."
        ),
    },
    "critical": {
        "label":       "⚠️ Critical Review",
        "description": "What's missing, vague, or needs clarification",
        "temperature": 0.4,
        "system": (
            "You are a critical reviewer and risk analyst. "
            "First answer the question using the context. "
            "Then identify: (1) MISSING or vague information, "
            "(2) RISKS or ambiguities in the document, "
            "(3) what the reader should WATCH OUT for. "
            "Be constructive and base your review on the document only."
        ),
    },
}


def generate_answer(style_key: str, question: str, context: str) -> dict:
    cfg = PROMPTS[style_key]
    user_message = (
        f"Document Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    try:
        answer = call_groq(
            system_prompt=cfg["system"],
            user_message=user_message,
            temperature=cfg["temperature"],
            max_tokens=600,
        )
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    return {
        "style":       style_key,
        "label":       cfg["label"],
        "description": cfg["description"],
        "answer":      answer,
    }


# ── Main entry point ───────────────────────────────────────────────────────────
def answer_question(question: str) -> dict:
    docs    = retrieve_chunks(question, k=5)
    context = format_context(docs)

    sources = [
        {
            "chunk_index": i + 1,
            "page":     doc.metadata.get("page", "?"),
            "strategy": doc.metadata.get("chunking", "N/A"),
            "content":  doc.page_content,
        }
        for i, doc in enumerate(docs)
    ]

    answers = [
        generate_answer(style_key, question, context)
        for style_key in ["factual", "analytical", "summary", "critical"]
    ]

    return {
        "question":    question,
        "answers":     answers,
        "sources":     sources,
        "chunks_used": len(docs),
    }