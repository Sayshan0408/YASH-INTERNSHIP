import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ── Cached embedding model (loaded once per process) ──────────────────────────
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("[EMBEDDINGS] Loading embedding model...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("[EMBEDDINGS] Embedding model loaded and cached.")
    return _embeddings


def step1_load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print("[STEP 1] Pages loaded: " + str(len(documents)))
    return documents


def step2_split_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print("[STEP 2] Chunks created: " + str(len(chunks)))
    return chunks


def step3_create_embeddings():
    embeddings = get_embeddings()
    print("[STEP 3] Embedding model ready")
    return embeddings


def step4_store_in_faiss(chunks, embeddings, index_path="faiss_index"):
    os.makedirs(index_path, exist_ok=True)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print("[STEP 4] Stored " + str(vectorstore.index.ntotal) + " vectors")
    return vectorstore


def ingest_pdf(pdf_path, index_path="faiss_index"):
    documents = step1_load_pdf(pdf_path)
    chunks = step2_split_chunks(documents)
    embeddings = step3_create_embeddings()
    vectorstore = step4_store_in_faiss(chunks, embeddings, index_path)
    return vectorstore, len(chunks)