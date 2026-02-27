import re
import numpy as np
import pdfplumber
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

_embed_model = None

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def extract_text_from_pdf(uploaded_file) -> str:
    full_text = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = re.sub(r'[ \t]+', ' ', text)
                text = text.strip()
                full_text.append(text)
    combined = "\n\n".join(full_text)
    if not combined.strip():
        raise ValueError("No text extracted. PDF may be a scanned image — use an OCR tool first.")
    return combined


def split_into_sentences(text: str) -> list[str]:
    ABBREVS = [
        r'Mr\.', r'Mrs\.', r'Ms\.', r'Dr\.', r'Prof\.', r'Sr\.', r'Jr\.',
        r'St\.', r'Ave\.', r'Blvd\.', r'Dept\.', r'Est\.', r'Fig\.', r'Vol\.',
        r'No\.', r'pp\.', r'etc\.', r'vs\.', r'e\.g\.', r'i\.e\.', r'approx\.',
        r'cf\.', r'al\.', r'et\s+al\.',
    ]
    protected    = text
    placeholders = {}

    # FIX 1: Correctly store the display form of each abbreviation
    # Original code used .replace(r'\.', '.') on a raw string — which never matched
    # because r'\.' is a two-character string, not a backslash-dot pattern.
    for i, abbrev in enumerate(ABBREVS):
        token = f"__ABBREV{i}__"
        # Convert the regex pattern back to a readable abbreviation for display
        display = re.sub(r'\\\.', '.', abbrev)       # \\. → .
        display = re.sub(r'\\s\+', ' ', display)     # \s+ → space (for "et al.")
        placeholders[token] = display
        protected = re.sub(abbrev, token, protected, flags=re.IGNORECASE)

    protected = re.sub(r'(\d)\.(\d)', r'\1__DECIMAL__\2', protected)
    raw = re.split(r'(?<=[.!?])\s+', protected)

    sentences = []
    for s in raw:
        s = s.strip()
        s = s.replace('__DECIMAL__', '.')
        for token, original in placeholders.items():
            s = s.replace(token, original)
        if len(s) >= 10 and re.search(r'[a-zA-Z]', s):
            sentences.append(s)
    return sentences


def sequential_chunk(text: str, chunk_size: int = 5) -> list[dict]:
    sentences = split_into_sentences(text)
    chunks    = []
    for i in range(0, len(sentences), chunk_size):
        group = sentences[i : i + chunk_size]
        chunks.append({
            "chunk_id":       len(chunks) + 1,
            "step":           len(chunks) + 1,
            "label":          f"Step {len(chunks) + 1}",
            "text":           " ".join(group),
            "sentences":      group,
            "sentence_range": f"{i+1}-{min(i+chunk_size, len(sentences))}",
            "type":           "sequential",
        })
    return chunks


def hierarchical_chunk(text: str) -> list[dict]:
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]
    chunks     = []
    for para_idx, para in enumerate(paragraphs):
        sentences = split_into_sentences(para)
        if not sentences:
            continue

        topic = sentences[0]

        # FIX 2: Correct level assignment for 2-sentence paragraphs.
        # Original: when len==2, supporting=[] and detail=sentences[1],
        # producing an orphaned Level 3 child with no Level 2 parent.
        # Fix: treat the second sentence as supporting (Level 2), not detail (Level 3).
        if len(sentences) == 1:
            supporting = []
            detail     = ""
        elif len(sentences) == 2:
            supporting = [sentences[1]]   # Level 2 — no orphaned Level 3
            detail     = ""
        else:
            supporting = sentences[1:-1]  # Level 2
            detail     = sentences[-1]    # Level 3

        chunk = {
            "paragraph": para_idx + 1,
            "chunk_id":  len(chunks) + 1,
            "level":     1,
            "label":     f"Main Topic {para_idx + 1}",
            "text":      para,
            "topic":     topic,
            "type":      "hierarchical",
            "children":  [],
        }
        if supporting:
            chunk["children"].append({
                "level": 2,
                "label": f"Supporting {para_idx + 1}",
                "text":  " ".join(supporting),
                "type":  "hierarchical",
            })
        if detail and detail != topic:
            chunk["children"].append({
                "level": 3,
                "label": f"Detail {para_idx + 1}",
                "text":  detail,
                "type":  "hierarchical",
            })
        chunks.append(chunk)
    return chunks


def flatten_hierarchical(chunks: list[dict]) -> list[dict]:
    flat = []
    for chunk in chunks:
        parent = {k: v for k, v in chunk.items() if k != "children"}
        flat.append(parent)
        for child in chunk.get("children", []):
            flat.append(child)
    return flat


def semantic_chunk(text: str, threshold: float = 0.5) -> list[dict]:
    """
    Splits text into semantically coherent chunks by detecting where meaning
    shifts between consecutive sentences.

    Core logic:
      1. Embed all sentences with a sentence-transformer model.
      2. Compute cosine similarity between each consecutive pair.
      3. Detect a 'split point' where similarity drops below an
         adaptive threshold — meaning the topic has changed.
      4. Group sentences between split points into one chunk.

    FIX 3 — Adaptive threshold was too aggressive.
    Original: effective_threshold = min(threshold, p25 + 0.05)
    Problem:  For homogeneous documents (high p25), this clamps the threshold
              so high that almost nothing triggers a split → one giant chunk.
              For very heterogeneous docs (low p25), p25+0.05 might be below 0,
              producing far too many splits.
    Fix:      Clamp the adaptive threshold between a hard floor (0.2) and the
              user-supplied ceiling, so both extremes behave correctly.

    FIX 4 — Similarity direction was wrong.
    Original: split when sim < threshold  (split on LOW similarity)
    This is semantically correct — but the grouping logic added the NEXT sentence
    to the current chunk before checking, causing off-by-one boundary errors.
    Fix:      Check similarity BEFORE appending, then decide whether to continue
              the current chunk or close it and start a new one.
    """
    if not text or not text.strip():
        raise ValueError("Input text is empty.")

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    if len(sentences) == 1:
        return [{
            "chunk_id":       1,
            "text":           sentences[0],
            "sentences":      sentences,
            "sentence_count": 1,
            "avg_similarity": 1.0,
            "split_score":    None,
            "label":          "Semantic Chunk 1",
            "type":           "semantic",
        }]

    model      = get_embed_model()
    embeddings = model.encode(sentences, show_progress_bar=False)

    # L2-normalise so dot product == cosine similarity
    norms  = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms  = np.where(norms == 0, 1e-10, norms)
    normed = embeddings / norms

    # Similarity between sentence i and sentence i+1
    similarities = np.sum(normed[:-1] * normed[1:], axis=1).tolist()

    # FIX 3: Adaptive threshold with both a floor and a ceiling
    p25 = float(np.percentile(similarities, 25))
    p75 = float(np.percentile(similarities, 75))
    # Use the midpoint of the lower half of the distribution as the split signal,
    # but never go below 0.20 (always split something) or above the user threshold.
    adaptive = (p25 + p75) / 2          # natural midpoint of similarity spread
    effective_threshold = float(np.clip(adaptive, 0.20, threshold))

    # FIX 4: Correct boundary — check similarity BEFORE deciding to extend chunk
    chunks        = []
    current_sents = [sentences[0]]
    current_sims  = []

    for i, sim in enumerate(similarities):
        next_sentence = sentences[i + 1]

        if sim >= effective_threshold:
            # Sentences are semantically similar → stay in the same chunk
            current_sents.append(next_sentence)
            current_sims.append(sim)
        else:
            # Similarity dropped → topic changed, close the current chunk
            chunks.append({
                "chunk_id":       len(chunks) + 1,
                "text":           " ".join(current_sents),
                "sentences":      current_sents.copy(),
                "sentence_count": len(current_sents),
                "avg_similarity": float(np.mean(current_sims)) if current_sims else 1.0,
                "split_score":    round(sim, 4),   # the similarity that triggered the split
                "label":          f"Semantic Chunk {len(chunks) + 1}",
                "type":           "semantic",
            })
            # Start a fresh chunk with the next sentence
            current_sents = [next_sentence]
            current_sims  = []

    # Flush the last chunk
    if current_sents:
        chunks.append({
            "chunk_id":       len(chunks) + 1,
            "text":           " ".join(current_sents),
            "sentences":      current_sents.copy(),
            "sentence_count": len(current_sents),
            "avg_similarity": float(np.mean(current_sims)) if current_sims else 1.0,
            "split_score":    None,
            "label":          f"Semantic Chunk {len(chunks) + 1}",
            "type":           "semantic",
        })

    return chunks


def parallel_chunk(text: str, num_aspects: int = 4) -> list[dict]:
    """
    Groups sentences by topic using KMeans clustering on embeddings.

    FIX 5 — Sentence order was not preserved within each cluster.
    Original: sentences appended in cluster-label order (i.e. arbitrary).
    Fix:      Track each sentence's original index and sort within clusters
              so the text reads in document order.
    """
    if not text or not text.strip():
        raise ValueError("Input text is empty.")

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    actual_k = min(num_aspects, len(sentences))
    if actual_k == 1:
        return [{
            "aspect":    1,
            "chunk_id":  1,
            "label":     "Aspect 1: full document",
            "text":      " ".join(sentences),
            "sentences": sentences,
            "type":      "parallel",
        }]

    model      = get_embed_model()
    embeddings = model.encode(sentences, show_progress_bar=False)
    kmeans     = KMeans(n_clusters=actual_k, n_init=10, random_state=42)
    labels     = kmeans.fit_predict(embeddings)

    # FIX 5: Store (original_index, sentence) pairs to preserve document order
    clusters: dict[int, list[tuple[int, str]]] = {k: [] for k in range(actual_k)}
    for orig_idx, (sentence, label) in enumerate(zip(sentences, labels)):
        clusters[label].append((orig_idx, sentence))

    chunks = []
    for cluster_id in range(actual_k):
        indexed_sents = clusters[cluster_id]
        if not indexed_sents:
            continue

        # Sort by original document position
        indexed_sents.sort(key=lambda x: x[0])
        cluster_sentences = [s for _, s in indexed_sents]
        orig_indices      = [i for i, _ in indexed_sents]

        centroid       = kmeans.cluster_centers_[cluster_id]
        cluster_embeds = embeddings[orig_indices]
        sims           = cosine_similarity(cluster_embeds, centroid.reshape(1, -1)).flatten()
        best_local_idx = int(np.argmax(sims))
        representative = cluster_sentences[best_local_idx]
        cohesion       = float(np.mean(sims))
        keywords       = _extract_keywords(" ".join(cluster_sentences))

        chunks.append({
            "aspect":         cluster_id + 1,
            "chunk_id":       cluster_id + 1,
            "label":          f"Aspect {cluster_id + 1}: {keywords}",
            "text":           " ".join(cluster_sentences),
            "sentences":      cluster_sentences,
            "representative": representative,
            "cohesion":       round(cohesion, 3),
            "type":           "parallel",
        })

    # Re-rank by cohesion (tightest cluster first)
    chunks.sort(key=lambda x: x["cohesion"], reverse=True)
    for i, chunk in enumerate(chunks):
        chunk["aspect"]   = i + 1
        chunk["chunk_id"] = i + 1
        chunk["label"]    = f"Aspect {i + 1}: " + chunk["label"].split(": ", 1)[-1]

    return chunks


def _extract_keywords(text: str, n: int = 3) -> str:
    STOPWORDS = {
        "the","a","an","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","can","need","dare","ought",
        "to","of","in","for","on","with","at","by","from","as","or",
        "and","but","if","then","that","this","these","those","it",
        "its","we","they","he","she","you","i","my","our","their",
        "also","more","such","than","not","which","what","when","how",
        "where","who","whom","any","all","each","every","both","few",
    }
    words    = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    freq     = Counter(filtered)
    top      = [word for word, _ in freq.most_common(n)]
    return ", ".join(top) if top else "general"


def chunk_text(text: str, method: str = "semantic", **kwargs) -> list[dict]:
    if not text or not text.strip():
        raise ValueError("Input text is empty.")
    method = method.lower().strip()
    if method == "sequential":
        return sequential_chunk(text, chunk_size=kwargs.get("chunk_size", 5))
    elif method == "hierarchical":
        return flatten_hierarchical(hierarchical_chunk(text))
    elif method == "semantic":
        return semantic_chunk(text, threshold=kwargs.get("threshold", 0.5))
    elif method == "parallel":
        return parallel_chunk(text, num_aspects=kwargs.get("num_aspects", 4))
    else:
        raise ValueError(f"Unknown method '{method}'. Valid: sequential | hierarchical | semantic | parallel")


def get_chunks_as_text(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        label = chunk.get("label", f"Chunk {chunk.get('chunk_id', '?')}")
        text  = chunk.get("text", "").strip()
        if text:
            parts.append(f"[{label}]\n{text}")
    return "\n\n".join(parts)


def get_chunk_stats(chunks: list[dict]) -> dict:
    if not chunks:
        return {"total_chunks": 0, "avg_words": 0, "min_words": 0,
                "max_words": 0, "total_words": 0, "chunk_type": "none"}
    word_counts = [len(c.get("text", "").split()) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_words":    round(sum(word_counts) / len(word_counts), 1),
        "min_words":    min(word_counts),
        "max_words":    max(word_counts),
        "total_words":  sum(word_counts),
        "chunk_type":   chunks[0].get("type", "unknown"),
    }