"""
streamlit_app.py  —  Web UI
─────────────────────────────────────────────────────────────────────────────
WHAT THIS FILE DOES:
  Builds a complete web interface that lets users:
    1. Upload a PDF / TXT / DOCX document
    2. Ask questions about it in plain English
    3. See the AI-generated answer with citations highlighted

  It talks to our FastAPI backend (main.py) over HTTP — just like
  any website talks to a backend server.

THINK OF IT LIKE:
  The "front of house" of a restaurant — what the customer sees and
  interacts with. The kitchen (FastAPI + RAG) works behind the scenes.

HOW TO RUN:
  streamlit run app/frontend/streamlit_app.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import time
import requests                # For making HTTP calls to our FastAPI backend
import streamlit as st         # The UI framework — every st.* call draws something
from pathlib import Path

# ── API Configuration ─────────────────────────────────────────────────────────
# os.getenv() reads the API_BASE_URL environment variable.
# Locally:        falls back to "http://localhost:8000"
# Inside Docker:  docker-compose sets API_BASE_URL=http://backend:8000
#                 so the frontend talks to the "backend" service by name
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


# ── Page Configuration ────────────────────────────────────────────────────────
# This MUST be the first Streamlit command in the file.
# It sets the browser tab title, icon, and layout.
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide",                  # Use the full browser width
    initial_sidebar_state="expanded",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────
# st.markdown() with unsafe_allow_html=True lets us inject raw CSS.
# This makes our app look polished and professional.
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #0284c7;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        font-size: 1.05rem;
        line-height: 1.7;
        color: #1e3a5f;
    }

    /* Citation card */
    .citation-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .citation-source {
        font-weight: 600;
        color: #0284c7;
    }
    .citation-snippet {
        color: #64748b;
        font-style: italic;
        margin-top: 0.25rem;
    }

    /* Status badge */
    .status-ready {
        background: #dcfce7;
        color: #166534;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-waiting {
        background: #fef9c3;
        color: #854d0e;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS  (talk to the FastAPI backend)
# ══════════════════════════════════════════════════════════════════════════════

def check_api_health() -> bool:
    """
    WHAT IT DOES:
        Pings GET /health on the FastAPI backend.
        Returns True if the server is running, False if it's offline.

    Used by the sidebar to show a green/red status indicator.
    """
    try:
        # timeout=3 means: give up after 3 seconds if no response
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        # Server is not running — requests raises this exception
        return False


def get_system_status() -> dict | None:
    """
    WHAT IT DOES:
        Calls GET /status to find out:
          - Are models loaded?
          - Is a document ready to query?
          - What document is currently loaded?

    Returns the JSON response as a Python dict, or None if the API is down.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        if response.status_code == 200:
            return response.json()    # .json() converts JSON string → Python dict
    except requests.exceptions.ConnectionError:
        pass
    return None


def upload_document(file_bytes: bytes, filename: str) -> dict | None:
    """
    WHAT IT DOES:
        Sends the uploaded file to POST /upload on the FastAPI backend.

    HOW HTTP FILE UPLOAD WORKS:
        We send a "multipart/form-data" request — the same format
        your browser uses when you submit a file input form.

    PARAMETERS:
        file_bytes (bytes): The raw file content
        filename (str):     The original filename (e.g. "report.pdf")

    RETURNS:
        dict with {"message", "filename", "chunks_created", "status"}
        or None if the upload failed
    """
    try:
        # "files" param tells requests to send as multipart/form-data
        # The tuple format is: (filename, file_content, content_type)
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
            timeout=120,    # Give up to 2 minutes for large documents
        )
        if response.status_code == 200:
            return response.json()
        else:
            # Show the error message from the server
            st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the backend. Is the FastAPI server running?")
    except requests.exceptions.Timeout:
        st.error("Upload timed out. The document may be too large.")
    return None


def ask_question(question: str) -> dict | None:
    """
    WHAT IT DOES:
        Sends the user's question to POST /ask on the FastAPI backend.
        Returns the answer + citations as a Python dict.

    PARAMETERS:
        question (str): The user's question in plain English

    RETURNS:
        dict with {"question", "answer", "citations", "time_taken_seconds"}
        or None if the request failed
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": question},  # Sends as application/json
            timeout=60,                   # LLM can take up to 60s on CPU
        )
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = response.json().get("detail", "Unknown error")
            st.error(f"Error: {error_msg}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the backend. Is the FastAPI server running?")
    except requests.exceptions.Timeout:
        st.error("Request timed out. The model is taking too long.")
    return None


def reset_system() -> bool:
    """
    WHAT IT DOES:
        Calls DELETE /reset to clear the vector store.
        Returns True if successful.
    """
    try:
        response = requests.delete(f"{API_BASE_URL}/reset", timeout=10)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
# st.sidebar.* draws elements in the left sidebar panel.

with st.sidebar:
    st.title("⚙️ System Status")
    st.divider()

    # ── API Health Check ──────────────────────────────────────────────────────
    api_alive = check_api_health()

    if api_alive:
        st.success("✅ API Server: Online")

        # Get detailed status from the backend
        sys_status = get_system_status()

        if sys_status:
            # Show model loading status
            if sys_status.get("models_loaded"):
                st.success("✅ AI Models: Loaded")
            else:
                st.warning("⏳ AI Models: Loading...")

            # Show document status
            if sys_status.get("document_loaded"):
                doc_name = sys_status.get("current_document", "Unknown")
                st.success(f"✅ Document: Ready")
                st.info(f"📄 {doc_name}")
            else:
                st.warning("⚠️ Document: Not loaded")
                st.caption("Upload a document to get started")
    else:
        st.error("❌ API Server: Offline")
        st.caption("Start the server with:")
        st.code("uvicorn app.backend.main:app --port 8000", language="bash")

    st.divider()

    # ── About section ─────────────────────────────────────────────────────────
    st.markdown("### 🔧 Tech Stack")
    st.markdown("""
    - 🔗 **LangChain** — RAG pipeline
    - 🤗 **Hugging Face** — Embeddings + LLM
    - ⚡ **FAISS** — Vector search
    - 🚀 **FastAPI** — Backend API
    - 🎈 **Streamlit** — This UI
    """)

    st.divider()

    # ── Reset button ──────────────────────────────────────────────────────────
    st.markdown("### 🗑️ Reset")
    if st.button("Clear Document & Index", type="secondary", use_container_width=True):
        # st.session_state stores data that persists between interactions
        if reset_system():
            # Clear our local UI state too
            st.session_state.pop("upload_success", None)
            st.session_state.pop("qa_history", None)
            st.success("System reset!")
            time.sleep(1)
            st.rerun()    # Refresh the whole UI
        else:
            st.error("Reset failed. Is the server running?")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📄 RAG Document Q&A System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload any document · Ask questions in plain English · '
    'Get answers with citations</p>',
    unsafe_allow_html=True
)

# ── Two column layout: Upload | Q&A ──────────────────────────────────────────
# st.columns([1, 1.5]) creates 2 columns.
# [1, 1.5] = left column is 1 unit wide, right is 1.5 units wide
col_upload, col_qa = st.columns([1, 1.5], gap="large")


# ─────────────────────────────────────────────────────────────────────────────
#  LEFT COLUMN: Document Upload
# ─────────────────────────────────────────────────────────────────────────────
with col_upload:
    st.markdown("### 📁 Upload Document")

    # st.file_uploader creates a drag-and-drop file upload widget
    # type= restricts which file types are accepted
    uploaded_file = st.file_uploader(
        label="Choose a file",
        type=["pdf", "txt", "docx"],
        help="Supported formats: PDF, TXT, DOCX. Max size: 200MB",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Show file details in a nice info box
        file_size_kb = len(uploaded_file.getvalue()) / 1024
        st.info(
            f"**{uploaded_file.name}**\n\n"
            f"Size: {file_size_kb:.1f} KB | "
            f"Type: {uploaded_file.type}"
        )

        # Upload button — only shown when a file is selected
        if st.button("⬆️ Process Document", type="primary", use_container_width=True):

            if not api_alive:
                st.error("Backend is offline. Please start the FastAPI server first.")
            else:
                # Show a spinner while uploading + ingesting
                with st.spinner("Uploading and processing document... (this may take 30–60s)"):
                    result = upload_document(
                        file_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                    )

                if result:
                    # Save success info to session state for display
                    st.session_state["upload_success"] = result
                    st.success(f"✅ Document processed successfully!")

    # Show upload results if we have them
    if "upload_success" in st.session_state:
        result = st.session_state["upload_success"]
        st.markdown("---")
        st.markdown("#### 📊 Ingestion Summary")

        # st.metric shows a big bold number with a label — great for dashboards
        m1, m2 = st.columns(2)
        m1.metric("Chunks Created", result.get("chunks_created", "?"))
        m2.metric("Status", result.get("status", "?").title())

        st.caption(f"File: `{result.get('filename', '?')}`")

    # Tips section
    st.markdown("---")
    st.markdown("#### 💡 Tips")
    st.markdown("""
    - Upload PDFs, text files, or Word docs
    - Shorter documents give faster, more accurate results
    - Each new upload **replaces** the previous one
    - Use the Reset button to clear and start fresh
    """)


# ─────────────────────────────────────────────────────────────────────────────
#  RIGHT COLUMN: Question & Answer
# ─────────────────────────────────────────────────────────────────────────────
with col_qa:
    st.markdown("### 💬 Ask a Question")

    # Initialize Q&A history in session state (persists across interactions)
    # session_state works like a dictionary that survives page re-runs
    if "qa_history" not in st.session_state:
        st.session_state["qa_history"] = []   # List of (question, answer, citations)

    # Check if a document is loaded before showing the input
    sys_status = get_system_status() if api_alive else None
    doc_loaded = sys_status.get("document_loaded", False) if sys_status else False

    if not doc_loaded:
        # Show a friendly placeholder message until a doc is uploaded
        st.info("👈 Please upload a document first to start asking questions.")
    else:
        # ── Question input form ───────────────────────────────────────────────
        # st.form groups widgets so they only submit together (avoids reruns on typing)
        with st.form(key="question_form", clear_on_submit=True):
            question_input = st.text_input(
                label="Your question",
                placeholder="e.g. What are the main challenges discussed in this document?",
                label_visibility="collapsed",
            )

            # Row with Ask button + example questions side by side
            btn_col, hint_col = st.columns([1, 2])
            with btn_col:
                submit = st.form_submit_button(
                    "🔍 Ask",
                    type="primary",
                    use_container_width=True,
                )
            with hint_col:
                st.caption("Press Enter or click Ask")

        # ── Handle form submission ────────────────────────────────────────────
        if submit and question_input.strip():
            with st.spinner("Thinking... (searching document + generating answer)"):
                result = ask_question(question_input.strip())

            if result:
                # Prepend to history so newest appears at top
                st.session_state["qa_history"].insert(0, result)

        elif submit and not question_input.strip():
            st.warning("Please type a question before clicking Ask.")

        # ── Example questions ─────────────────────────────────────────────────
        st.markdown("**Try these example questions:**")
        examples = [
            "What is the main topic of this document?",
            "What challenges are mentioned?",
            "What solutions or recommendations are proposed?",
            "Summarize the key findings.",
        ]
        # Show example buttons in a 2x2 grid
        ex_cols = st.columns(2)
        for i, example in enumerate(examples):
            with ex_cols[i % 2]:
                # Clicking an example button asks that question directly
                if st.button(example, key=f"ex_{i}", use_container_width=True):
                    with st.spinner("Thinking..."):
                        result = ask_question(example)
                    if result:
                        st.session_state["qa_history"].insert(0, result)
                        st.rerun()

    # ── Q&A History ───────────────────────────────────────────────────────────
    if st.session_state.get("qa_history"):
        st.markdown("---")
        st.markdown("### 📝 Q&A History")

        for i, entry in enumerate(st.session_state["qa_history"]):
            # st.expander creates a collapsible section
            # The first result is expanded by default
            with st.expander(
                f"Q: {entry['question'][:80]}{'...' if len(entry['question']) > 80 else ''}",
                expanded=(i == 0),
            ):
                # ── Answer ────────────────────────────────────────────────────
                st.markdown("**Answer:**")
                st.markdown(
                    f'<div class="answer-box">{entry["answer"]}</div>',
                    unsafe_allow_html=True,
                )

                # ── Response time metric ──────────────────────────────────────
                time_taken = entry.get("time_taken_seconds", 0)
                st.caption(f"⏱️ Response time: {time_taken:.2f}s")

                # ── Citations ─────────────────────────────────────────────────
                citations = entry.get("citations", [])
                if citations:
                    st.markdown("**📎 Sources used:**")
                    for citation in citations:
                        st.markdown(
                            f"""<div class="citation-card">
                                <span class="citation-source">
                                    📄 {citation['source']} — Page {citation['page']}
                                </span>
                                <div class="citation-snippet">
                                    "{citation['snippet'][:180]}..."
                                </div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No specific citations found for this answer.")

        # Button to clear history
        if st.button("🗑️ Clear History", key="clear_history"):
            st.session_state["qa_history"] = []
            st.rerun()
