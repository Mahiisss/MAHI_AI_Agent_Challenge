import os
import json
import re
from pathlib import Path
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss

# ---------------- CONFIG ----------------
MODEL_NAME =  r"C:\Users\ritik\Desktop\MAHI_AI_Agent_Challenge\models\all-MiniLM-L6-v2"
  # Local model path
CHUNK_SIZE = 500
OVERLAP = 100
EMB_DIM = 384

DATA_DIR = os.path.join(os.path.dirname(__file__), "storage")
os.makedirs(DATA_DIR, exist_ok=True)

META_PATH = os.path.join(DATA_DIR, "meta.json")

# ---------------- GLOBALS ----------------
model = SentenceTransformer(MODEL_NAME)
index = faiss.IndexFlatIP(EMB_DIM)

# Load metadata if exists
if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
else:
    meta = {"docs": []}


# ---------------- HELPERS ----------------
def extract_and_chunk_pdf(path: str, max_pages: int = 200):
    """Extract text page by page and split into chunks."""
    all_chunks = []
    path = Path(path)
    if not path.exists():
        return all_chunks

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            if i >= max_pages:
                break
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue

            start = 0
            while start < len(text):
                end = start + CHUNK_SIZE
                chunk = text[start:end]

                # avoid cutting in middle of a word
                if end < len(text):
                    extra_end = min(len(text), end + 50)
                    while end < extra_end and not text[end].isspace():
                        end += 1
                    chunk = text[start:end]

                all_chunks.append(chunk.strip())
                start = end - OVERLAP

            if len(all_chunks) > 2000:
                break

    return all_chunks


def add_document(doc_id: str, pdf_path: str):
    """Extract chunks, encode, store in FAISS + metadata."""
    global index, meta
    chunks = extract_and_chunk_pdf(pdf_path)
    if not chunks:
        print(f"[WARN] No chunks extracted from {pdf_path}")
        return 0

    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    for i, chunk in enumerate(chunks):
        meta["docs"].append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_{i}",
            "text": chunk
        })

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return len(chunks)


# ---------------- FIELD EXTRACTION ----------------
def _extract_field_from_text(question: str, text: str):
    """Regex extractor for fields like Name, Semester, CGPA, Email, Phone, GitHub."""
    q = (question or "").lower()
    t = (text or "").replace("\n", " ")

    # --- Name ---
    if "name" in q:
        match = re.search(r"(?:name of student|name)\s*[:\-]?\s*([A-Z][a-zA-Z ,.'-]{2,})", t, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # --- Semester ---
    if "semester" in q or "sem" in q:
        match = re.search(
            r"(?:semester|sem)\s*[:\-]?\s*([0-9]{1,2}|[ivxlcdm]+|[0-9]{1,2}(st|nd|rd|th)?)",
            t, re.IGNORECASE
        )
        if match:
            return match.group(1).strip()

    # --- GPA / CGPA / CPI / SGPA ---
    if any(word in q for word in ["gpa", "cgpa", "cpi", "sgpa"]):
        # Pattern 1: GPA: 7.05 / CGPA - 8.2
        match = re.search(r"(?:CGPA|GPA|CPI|SGPA)\s*[:=\-]?\s*([0-9]{1,2}(?:\.\d{1,4})?)", t, re.IGNORECASE)
        if match:
            return match.group(1)

        # Pattern 2: 7.05/10
        match = re.search(r"\b([0-9]{1,2}(?:\.\d{1,4})?)\s*/\s*10\b", t)
        if match:
            return match.group(1)

        # Pattern 3: 7.05 CGPA
        match = re.search(r"\b([0-9]{1,2}(?:\.\d{1,4})?)\s*(?:CGPA|GPA|CPI|SGPA)", t, re.IGNORECASE)
        if match:
            return match.group(1)

    # --- Email ---
    if "email" in q:
        m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", t)
        if m:
            return m.group(0)

    # --- Phone ---
    if "phone" in q or "contact" in q or "mobile" in q:
        m = re.search(r"(?:\+?\d{1,3}[\s-]?)?(?:\d[\d\s-]{7,}\d)", t)
        if m:
            return m.group(0).strip()

    # --- GitHub ---
    if "github" in q:
        m = re.search(r"https?://github\.com/[A-Za-z0-9_.-]+", t, re.IGNORECASE)
        if m:
            return m.group(0)

    return None


# ---------------- FULL-DOC EXTRACTION ----------------
def extract_from_full_doc(question: str, doc_id: str):
    """Force regex extraction from the entire document text, not just chunks."""
    texts = [d["text"] for d in meta["docs"] if d["doc_id"] == doc_id]
    if not texts:
        return None
    full_text = " ".join(texts)
    return _extract_field_from_text(question, full_text)


# ---------------- QUERY ----------------
def query(question: str, k: int = 5):
    """First try full-doc regex extraction, then semantic search fallback."""
    if meta["docs"]:
        doc_id = meta["docs"][-1]["doc_id"]  # latest doc uploaded
        forced = extract_from_full_doc(question, doc_id)
        if forced:
            return [{
                "score": 1.0,
                "answer": forced,
                "context": "Extracted directly from document",
                "doc_id": doc_id,
                "chunk_id": "full_doc"
            }]

    if index.ntotal == 0:
        return []

    q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(meta["docs"]):
            d = meta["docs"][idx]
            text_chunk = d["text"]
            answer = _extract_field_from_text(question, text_chunk)
            results.append({
                "score": float(score),
                "answer": answer if answer else text_chunk[:200],
                "context": text_chunk[:500],
                "doc_id": d["doc_id"],
                "chunk_id": d["chunk_id"]
            })
    return results


# ---------------- SUMMARY ----------------
def summarize_doc(doc_id: str, word_count: int = 120):
    """Summarize text for a doc. Prepend CGPA if found."""
    texts = [d["text"] for d in meta["docs"] if d["doc_id"] == doc_id]
    if not texts:
        return "No text found for this document."

    combined = " ".join(texts)
    words = combined.split()

    cgpa_match = re.search(r"(?:CGPA|GPA|SGPA|CPI)\s*[:=\-]?\s*([0-9]+(?:\.\d{1,4})?)", combined, re.IGNORECASE)
    cgpa_line = f"📊 CGPA: {cgpa_match.group(1)}\n" if cgpa_match else ""

    summary_text = " ".join(words[:word_count]) + (" ..." if len(words) > word_count else "")
    return cgpa_line + summary_text
