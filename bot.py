import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from huggingface_hub import InferenceClient
from typing import Optional, List, Any

load_dotenv()


# ── Custom LLM using chat_completion (works with conversational models) ────────
class HFChatLLM(LLM):
    repo_id: str
    hf_token: str
    max_new_tokens: int = 512
    temperature: float = 0.2

    @property
    def _llm_type(self) -> str:
        return "huggingface_chat"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        client = InferenceClient(model=self.repo_id, token=self.hf_token)
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()


# ── Re-use cached embedding model from ingest ─────────────────────────────────
def get_embeddings():
    from ingest import get_embeddings as _get
    return _get()


def step5_load_vectorstore(index_path="faiss_index"):
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    print("[STEP 5] FAISS index loaded from: " + index_path)
    return vectorstore


def step6_create_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )


def step7_build_chain(retriever):
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HUGGINGFACEHUB_API_TOKEN is not set. "
            "Please add it to your .env file."
        )

    llm = HFChatLLM(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        hf_token=hf_token,
        max_new_tokens=512,
        temperature=0.2,
    )

    prompt = PromptTemplate(
        template=(
            "You are an expert document assistant. "
            "Use ONLY the context below to answer. "
            "If the answer is not found, say: "
            "I cannot find this in the document.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    print("[STEP 7] RetrievalQA chain built successfully.")
    return chain


def load_chain(index_path="faiss_index"):
    vectorstore = step5_load_vectorstore(index_path)
    retriever = step6_create_retriever(vectorstore)
    chain = step7_build_chain(retriever)
    return chain


# ── Summary Function ───────────────────────────────────────────────────────────
def summarize_answer(question: str, answer: str, sources: list) -> str:
    """
    Takes the original question, the full answer, and source chunks,
    then generates a concise, accurate bullet-point summary using Qwen.
    """
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise EnvironmentError("HUGGINGFACEHUB_API_TOKEN is not set.")

    # Build context from source chunks for extra accuracy
    source_context = ""
    for i, doc in enumerate(sources):
        page = doc.metadata.get("page", "?")
        source_context += f"\n[Chunk {i+1} - Page {page}]:\n{doc.page_content}\n"

    prompt = (
        "You are a precise document summarizer.\n\n"
        "A user asked the following question and received an answer from a document. "
        "Your task is to produce a SHORT, ACCURATE summary of the answer specifically "
        "related to the question. Use bullet points. Be concise. "
        "Do NOT add any information not present in the answer or source chunks.\n\n"
        f"Question Asked:\n{question}\n\n"
        f"Full Answer:\n{answer}\n\n"
        f"Supporting Source Chunks:{source_context}\n\n"
        "Now write a clear bullet-point summary of the answer to the question above. "
        "Each bullet should be a key point directly relevant to the question. "
        "Keep it under 6 bullets. Start each bullet with '•'.\n\n"
        "Summary:"
    )

    client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=hf_token)
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.1,  # Very low temp for accuracy
    )
    return response.choices[0].message.content.strip()