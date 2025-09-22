from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os, uuid
from processor import add_document, query, summarize_doc

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="📄 Document Analyzer Agent ")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend anywhere
    allow_methods=["*"],
    allow_headers=["*"]
)


# ---------------- Upload Endpoint ----------------
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    uid = str(uuid.uuid4())[:8]
    fname = f"{uid}_{file.filename}"
    path = os.path.join(UPLOAD_DIR, fname)

    with open(path, "wb") as f:
        f.write(content)

    chunks = add_document(uid, path)

    return {
        "doc_id": uid,
        "filename": file.filename,
        "chunks_added": chunks
    }


# ---------------- Ask Question Endpoint ----------------
@app.post("/ask")
async def ask(question: str = Form(...), top_k: int = Form(5)):
    results = query(question, k=top_k)
    return {
        "question": question,
        "results": results
    }


# ---------------- Summarize Endpoint ----------------
@app.post("/summary")
async def summary(doc_id: str = Form(...), word_count: int = Form(120)):
    s = summarize_doc(doc_id, word_count=word_count)
    return {
        "doc_id": doc_id,
        "summary": s
    }
