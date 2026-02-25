"""
ingest.py — PDF ingestion with 4 chunking strategies + best-pick selection.

Strategies:
  1. Recursive Character — fixed-size with overlap (baseline)
  2. Sentence-Level      — groups of N sentences per chunk
  3. Semantic/Paragraph  — paragraph/topic-aware grouping
  4. Token-Aware         — splits on ~token budget (approx. 4 chars/token)

Best strategy is auto-selected by a coherence scoring heuristic.
"""

import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ── Cached embedding model ─────────────────────────────────────────────────────
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("[EMBEDDINGS] Loading all-MiniLM-L6-v2...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("[EMBEDDINGS] Ready.")
    return _embeddings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRATEGY 1 — Recursive Character Splitter
# Splits by fixed character count. Falls back through separators:
#   paragraph → newline → sentence → space
# Best for: mixed/structured documents
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chunk_recursive(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    raw = splitter.split_documents(documents)
    chunks = [
        Document(page_content=d.page_content,
                 metadata={**d.metadata, "chunking": "recursive"})
        for d in raw
    ]
    print(f"[CHUNKING] Strategy 1 — Recursive: {len(chunks)} chunks")
    return chunks, "recursive"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRATEGY 2 — Sentence-Level Chunking
# Splits text on sentence boundaries (. ! ?) then groups N sentences.
# Best for: narrative text, reports, proposals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chunk_sentences(documents, sentences_per_chunk: int = 5):
    results = []
    for doc in documents:
        text      = doc.page_content
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        for i in range(0, len(sentences), sentences_per_chunk):
            group = " ".join(sentences[i : i + sentences_per_chunk]).strip()
            if group:
                results.append(Document(
                    page_content=group,
                    metadata={**doc.metadata, "chunking": "sentence"}
                ))
    print(f"[CHUNKING] Strategy 2 — Sentence-Level: {len(results)} chunks")
    return results, "sentence"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRATEGY 3 — Semantic / Paragraph-Aware Chunking
# Respects paragraph breaks (double newlines) as natural topic boundaries.
# Long paragraphs are sub-split; very short ones are skipped as noise.
# Best for: formal documents, legal text, RFPs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chunk_semantic(documents):
    results = []
    sub_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    for doc in documents:
        text       = doc.page_content
        paragraphs = re.split(r'\n{2,}', text.strip())
        for para in paragraphs:
            para = para.strip()
            if len(para) < 40:
                continue  # skip noise / headers alone
            if len(para) > 800:
                for sub in sub_splitter.split_text(para):
                    results.append(Document(
                        page_content=sub,
                        metadata={**doc.metadata, "chunking": "semantic"}
                    ))
            else:
                results.append(Document(
                    page_content=para,
                    metadata={**doc.metadata, "chunking": "semantic"}
                ))
    print(f"[CHUNKING] Strategy 3 — Semantic/Paragraph: {len(results)} chunks")
    return results, "semantic"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRATEGY 4 — Token-Aware Chunking
# Approximates token count (1 token ≈ 4 chars) and targets a token budget.
# Splits at sentence boundaries closest to the token budget.
# Best for: LLM context window management, large documents
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def chunk_token_aware(documents, target_tokens: int = 150, overlap_tokens: int = 20):
    target_chars  = target_tokens * 4
    overlap_chars = overlap_tokens * 4
    results = []

    for doc in documents:
        text      = doc.page_content
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        current_chunk = []
        current_len   = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > target_chars and current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    results.append(Document(
                        page_content=chunk_text,
                        metadata={**doc.metadata, "chunking": "token-aware"}
                    ))
                # Overlap: keep last sentences that fit in overlap budget
                overlap_text = []
                overlap_acc  = 0
                for s in reversed(current_chunk):
                    if overlap_acc + len(s) <= overlap_chars:
                        overlap_text.insert(0, s)
                        overlap_acc += len(s)
                    else:
                        break
                current_chunk = overlap_text + [sent]
                current_len   = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sent)
                current_len += sent_len

        # Flush remainder
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                results.append(Document(
                    page_content=chunk_text,
                    metadata={**doc.metadata, "chunking": "token-aware"}
                ))

    print(f"[CHUNKING] Strategy 4 — Token-Aware: {len(results)} chunks")
    return results, "token-aware"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BEST-PICK SELECTOR
# Scores each strategy on:
#   - Sentence density (more coherent = better)
#   - Ideal length penalty (chunks should be 150–600 chars)
#   - Chunk count penalty (too few or too many is bad)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def score_chunks(chunks) -> float:
    if not chunks:
        return 0.0
    scores = []
    for c in chunks:
        text           = c.page_content
        length         = len(text)
        sentence_count = len(re.findall(r'[.!?]', text)) + 1
        density        = sentence_count / max(length, 1) * 1000
        length_score   = 1.0 if 150 <= length <= 600 else max(0, 1 - abs(length - 375) / 600)
        scores.append(density * length_score)
    avg = sum(scores) / len(scores)
    # Slight bonus for having a healthy number of chunks (not too few, not extreme)
    count_bonus = 1.0 if 10 <= len(chunks) <= 300 else 0.8
    return avg * count_bonus


def pick_best_strategy(documents):
    strategies = [
        chunk_recursive(documents),
        chunk_sentences(documents),
        chunk_semantic(documents),
        chunk_token_aware(documents),
    ]
    best_chunks, best_strategy = None, None
    best_score = -1
    for chunks, name in strategies:
        score = score_chunks(chunks)
        print(f"[SCORE] {name}: {score:.4f}")
        if score > best_score:
            best_score    = score
            best_chunks   = chunks
            best_strategy = name
    print(f"[BEST] Strategy selected: {best_strategy} (score={best_score:.4f})")
    return best_chunks, best_strategy


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN INGEST PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ingest_pdf(pdf_path: str, index_path: str = "faiss_index") -> dict:
    # Step 1: Load PDF
    loader    = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"[STEP 1] Loaded {len(documents)} pages from {pdf_path}")

    # Step 2: Auto-pick best chunking strategy
    chunks, strategy_used = pick_best_strategy(documents)

    # Step 3: Embed & store in FAISS
    embeddings = get_embeddings()
    os.makedirs(index_path, exist_ok=True)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"[STEP 4] Stored {vectorstore.index.ntotal} vectors to {index_path}")

    return {
        "chunk_count":   len(chunks),
        "strategy_used": strategy_used,
        "pages":         len(documents),
        "vectors":       vectorstore.index.ntotal
    }