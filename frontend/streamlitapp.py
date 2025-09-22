import streamlit as st
import requests

API = "http://127.0.0.1:8000"  # Backend FastAPI server

st.set_page_config(page_title="📄 Document Analyzer Agent — Ku", layout="wide")
st.title("📄 Document Analyzer Agent ")

# ---------------- 1) Upload PDF ----------------
st.header("1️⃣ Upload PDF")
uploaded_file = st.file_uploader("Upload PDF files", type="pdf")

if uploaded_file:
    if st.button("📤 Upload to Backend"):
        try:
            res = requests.post(
                f"{API}/upload_pdf",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            )
            if res.status_code == 200:
                data = res.json()
                st.success(
                    f"✅ Uploaded **{data.get('filename', uploaded_file.name)}** "
                    f"(doc_id: `{data.get('doc_id', 'N/A')}`, chunks: {data.get('chunks_added', 0)})"
                )
                st.session_state["doc_id"] = data.get("doc_id")
            else:
                st.error(f"❌ Upload failed: {res.text}")
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Backend not running. Start FastAPI server first!")

# ---------------- 2) Ask Question ----------------
st.header("2️⃣ Ask a Question")
if "doc_id" in st.session_state:
    q = st.text_input("Ask a question about the uploaded document (e.g., What is the CGPA?)")
    k = st.slider("Top K chunks (semantic search)", 1, 10, 4)
    if st.button("🔎 Ask") and q.strip():
        try:
            res = requests.post(f"{API}/ask", data={"question": q, "top_k": k})
            if res.status_code == 200:
                results = res.json().get("results", [])
                if results:
                    for r in results:
                        ans = r.get("answer", "⚠️ Not found")
                        ctx = r.get("context", "")
                        if "⚠️" not in ans and "Not found" not in ans:
                            st.success(f"📌 **Answer:** {ans}")
                        else:
                            st.warning("❓ Couldn’t find exact field. Closest context shown:")
                        st.caption(f"Context (Doc {r.get('doc_id')}, Score {r.get('score'):.3f}): {ctx[:300]}...")
                        st.markdown("---")
                else:
                    st.warning("⚠️ No matching results found.")
            else:
                st.error(f"❌ Query failed: {res.text}")
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Backend not running. Start FastAPI server first!")
else:
    st.info("📌 Upload a PDF first before asking a question.")

# ---------------- 3) Summarize Document ----------------
st.header("3️⃣ Summarize Document")
if "doc_id" in st.session_state:
    wc = st.slider("Word count for summary", 50, 300, 120)
    if st.button("📝 Summarize"):
        try:
            res = requests.post(f"{API}/summary", data={"doc_id": st.session_state['doc_id'], "word_count": wc})
            if res.status_code == 200:
                summary = res.json().get("summary", "⚠️ No summary returned.")
                st.subheader("📌 Document Summary")
                st.write(summary)
            else:
                st.error(f"❌ Summary failed: {res.text}")
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Backend not running. Start FastAPI server first!")
else:
    st.info("📌 Upload a PDF first before summarizing.")







