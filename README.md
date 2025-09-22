# MAHI_AI_Agent_Challenge

## Project: Document Analyzer Agent

### What it does
This agent extracts structured information (semester, CGPA/GPA, email, phone, GitHub), summarizes the document,
answers freeform questions by combining regex extraction and semantic search over document chunks.

### Key features
- Upload PDF and index it (embedding + FAISS)
- Field extraction: CGPA/GPA (multiple formats), semester (numbers/roman/ordinal), name, email, phone, GitHub.
- Ask natural language questions; returns exact field if found, otherwise best context snippet.
- Summary generator that highlights CGPA first.

### Tools & APIs used
- FastAPI (backend) + uvicorn
- Streamlit (frontend)
- pdfplumber (PDF text extraction)
- sentence-transformers (all-MiniLM-L6-v2) - local preferred
- faiss-cpu (semantic search)
- huggingface_hub (optional)

### Setup (local)


1. Create and activate venv (Windows PowerShell):
   ```powershell
   python -m venv .venv
   & ".\.venv\Scripts\Activate.ps1"
   pip install -r requirements.txt
   ```

2. OPTIONAL: If you face HF rate limits, download the model manually and update `processor.py` MODEL_NAME to your loc
   ```
   MODEL_NAME = r"C:\Users\ritik\Desktop\Mahi_AI_Agent\models\all-MiniLM-L6-v2"
   ```

3. Start backend:
   ```powershell
   cd backend
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

4. Start frontend (new terminal, with venv activated):
   ```powershell
   cd .. or cd frontend 
   streamlit run frontendapp.py
   ```

6. Use the UI at http://localhost:8501


### Limitations
- Works best with machine-generated PDFs (not OCR). If PDFs are scanned images, OCR step is required (not included).
- FAISS index is in-memory (not persisted) by default.
- You may need to set `MODEL_NAME` in `processor.py` to a local path to avoid Hugging Face rate limiting.


## Architecture
![Architecture Diagram](architecture.png)

### Folder structure
Ritik_AI_Agent_Challenge/
 ├── backend/
 │   ├── app.py           # FastAPI server
 │   ├── processor.py     # PDF processing & embeddings
 ├── frontend/
 │   └── frontend_app.py  # Streamlit frontend
 ├── requirements.txt     # Dependencies
 ├── architecture.png     # System architecture diagram
 ├── README.md            # Documentation
 └── gifs/                # Screenshots / demo GIFs (optional)




