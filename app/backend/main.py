"""
main.py  —  FastAPI Backend
─────────────────────────────────────────────────────────────────────────────
WHAT THIS FILE DOES:
  Turns our RAG system into a web server with 5 API endpoints:

  GET  /health    → "Is the server alive?"
  GET  /status    → "Is a document loaded and ready?"
  POST /upload    → "Here's a PDF — please ingest it"
  POST /ask       → "Here's my question — give me an answer + citations"
  DELETE /reset   → "Clear everything, start fresh"

THINK OF IT LIKE:
  A restaurant API:
    /upload  → "Here is today's menu (document)"
    /ask     → "I'd like to order (question)"
    /health  → "Is the kitchen open?"
    /reset   → "Clear the table"
─────────────────────────────────────────────────────────────────────────────
"""

import os
import shutil
import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager

# FastAPI: the web framework
from fastapi import FastAPI, File, UploadFile, HTTPException, status
# BaseModel: for defining the shape of request/response JSON bodies
from pydantic import BaseModel
# JSONResponse: lets us return custom JSON with specific status codes
from fastapi.responses import JSONResponse
# CORS: allows our Streamlit frontend (different port) to call this API
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

# Our own modules
from app.backend.ingestion import ingest_document
from app.backend.embeddings import (
    get_embedding_model,
    ingest_and_store,
    load_vector_store,
    vector_store_exists,
)
from app.backend.qa_chain import load_llm, answer_question

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
STORE_NAME = "main_index"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Global State ──────────────────────────────────────────────────────────────
# These variables live for the entire lifetime of the server.
# We load the heavy models ONCE at startup (not on every request).
#
# WHY GLOBAL? Loading a model takes 10–30 seconds. If we reloaded on
# every request, each question would take 30 seconds just to load!
app_state = {
    "embedding_model": None,   # Loaded once at startup
    "llm":             None,   # Loaded once at startup
    "vector_store":    None,   # Created/loaded when a doc is uploaded
    "document_loaded": False,  # Flag: is a document ready to query?
    "current_doc":     None,   # Name of the currently loaded document
}


# ── Lifespan: runs at startup and shutdown ────────────────────────────────────
# @asynccontextmanager turns this into a "context manager"
# FastAPI calls the code BEFORE yield at startup, and AFTER yield at shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load the embedding model and LLM into memory.
    Shutdown: clean up resources.

    WHY AT STARTUP?
        Both models are heavy (~80MB + ~250MB). Loading them once means
        every subsequent /ask request responds in < 2 seconds instead of 30.
    """
    logger.info("=" * 50)
    logger.info("  RAG Document Q&A API — Starting up...")
    logger.info("=" * 50)

    # Load embedding model (fast, ~3 seconds)
    logger.info("Loading embedding model...")
    app_state["embedding_model"] = get_embedding_model()

    # Load LLM (slower, ~20 seconds on first run)
    logger.info("Loading language model (this takes ~20s first time)...")
    app_state["llm"] = load_llm()

    # If a vector store was saved from a previous run, load it automatically
    if vector_store_exists(STORE_NAME):
        logger.info("Found existing vector store — loading it...")
        app_state["vector_store"] = load_vector_store(
            app_state["embedding_model"], STORE_NAME
        )
        app_state["document_loaded"] = True
        logger.info("Previous document index loaded and ready!")

    logger.info("Server ready! All systems go.")
    logger.info("=" * 50)

    yield  # ← Server runs here, handling requests

    # Shutdown cleanup (runs when you press Ctrl+C)
    logger.info("Server shutting down. Goodbye!")


# ── Create the FastAPI app ────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Document Q&A API",
    description=(
        "Upload documents and ask questions. "
        "Powered by LangChain, Hugging Face, and FAISS."
    ),
    version="1.0.0",
    lifespan=lifespan,  # Register our startup/shutdown logic
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
# CORS (Cross-Origin Resource Sharing): browsers block requests between
# different ports by default (e.g., Streamlit on :8501 calling API on :8000).
# This middleware tells the browser: "It's OK, allow all origins."
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow any origin (fine for local dev)
    allow_credentials=True,
    allow_methods=["*"],       # Allow GET, POST, DELETE, etc.
    allow_headers=["*"],       # Allow any headers
)


# ── Request / Response Models ─────────────────────────────────────────────────
# Pydantic models define the SHAPE of JSON bodies.
# FastAPI auto-validates incoming data against these models.

class QuestionRequest(BaseModel):
    """What the /ask endpoint expects to receive."""
    question: str                   # The user's question text

class Citation(BaseModel):
    """A single citation entry."""
    source: str                     # Filename e.g. "report.pdf"
    page: int                       # Page number (1-indexed)
    snippet: str                    # Short preview of the relevant text

class AnswerResponse(BaseModel):
    """What the /ask endpoint sends back."""
    question:  str
    answer:    str
    citations: list[Citation]
    time_taken_seconds: float       # How long the Q&A took (for perf tracking)

class StatusResponse(BaseModel):
    """What the /status endpoint sends back."""
    document_loaded: bool
    current_document: str | None
    vector_store_exists: bool
    models_loaded: bool


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
async def health_check():
    """
    ENDPOINT: GET /health
    PURPOSE:  Quick check — is the server alive?
    RETURNS:  {"status": "ok"}

    Used by Docker, monitoring tools, and the Streamlit UI to verify
    the backend is running before making other requests.
    """
    return {"status": "ok", "message": "RAG Q&A API is running"}


@app.get("/status", response_model=StatusResponse, tags=["System"])
async def get_status():
    """
    ENDPOINT: GET /status
    PURPOSE:  Returns detailed state of the system.
    RETURNS:  Whether models are loaded, whether a doc is ready, etc.
    """
    return StatusResponse(
        document_loaded=app_state["document_loaded"],
        current_document=app_state["current_doc"],
        vector_store_exists=vector_store_exists(STORE_NAME),
        models_loaded=(
            app_state["embedding_model"] is not None
            and app_state["llm"] is not None
        ),
    )


@app.post("/upload", tags=["Document"])
async def upload_document(file: UploadFile = File(...)):
    """
    ENDPOINT: POST /upload
    PURPOSE:  Accept a PDF/TXT/DOCX upload, ingest it, build FAISS index.

    HOW IT WORKS:
      1. Receive the file bytes from the HTTP request
      2. Save the file to data/uploads/
      3. Run ingestion (load + chunk)
      4. Run embeddings (chunk → vector → FAISS)
      5. Save index to disk
      6. Update app_state so /ask knows a doc is ready

    UploadFile is FastAPI's built-in file type — it handles
    streaming large files efficiently without loading all into RAM.
    """
    # ── Validate file type ────────────────────────────────────────────────────
    allowed_extensions = {".pdf", ".txt", ".docx"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        # 400 Bad Request — user sent an unsupported file type
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{file_ext}' not supported. Use: .pdf, .txt, .docx"
        )

    # ── Save file to disk ─────────────────────────────────────────────────────
    save_path = UPLOAD_DIR / file.filename
    logger.info(f"Receiving file: {file.filename}")

    try:
        # Open a file on disk and write the uploaded bytes into it
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved to: {save_path}")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )

    # ── Ingest the document ───────────────────────────────────────────────────
    try:
        logger.info("Ingesting document...")
        chunks = ingest_document(str(save_path))
        logger.info(f"Document split into {len(chunks)} chunks")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process document: {str(e)}"
        )

    # ── Create FAISS vector store ─────────────────────────────────────────────
    try:
        logger.info("Building vector store...")
        vector_store = ingest_and_store(
            chunks,
            store_name=STORE_NAME,
        )
        # Update global state — the system is now ready to answer questions
        app_state["vector_store"]    = vector_store
        app_state["document_loaded"] = True
        app_state["current_doc"]     = file.filename
        logger.info("Vector store ready!")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build vector store: {str(e)}"
        )

    return {
        "message":      "Document ingested successfully!",
        "filename":     file.filename,
        "chunks_created": len(chunks),
        "status":       "ready",
    }


@app.post("/ask", response_model=AnswerResponse, tags=["Q&A"])
async def ask_question(request: QuestionRequest):
    """
    ENDPOINT: POST /ask
    PURPOSE:  Answer a question using the loaded document.

    HOW IT WORKS:
      1. Validate a document is loaded (else 400 error)
      2. Run the full RAG pipeline (retrieve → prompt → LLM → answer)
      3. Return answer + citations + time taken

    REQUEST BODY (JSON):
      {"question": "What are the challenges of AI in healthcare?"}

    RESPONSE BODY (JSON):
      {
        "question": "What are the challenges...",
        "answer": "The main challenges are...",
        "citations": [{"source": "doc.pdf", "page": 3, "snippet": "..."}],
        "time_taken_seconds": 1.87
      }
    """
    # Guard: make sure a document has been uploaded and indexed first
    if not app_state["document_loaded"] or app_state["vector_store"] is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document loaded. Please upload a document first via POST /upload"
        )

    # Guard: make sure the question isn't empty
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )

    # ── Run the RAG pipeline ──────────────────────────────────────────────────
    start_time = time.time()

    try:
        result = answer_question(
            question=request.question,
            vector_store=app_state["vector_store"],
            llm=app_state["llm"],
        )
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {str(e)}"
        )

    elapsed = round(time.time() - start_time, 2)

    # ── Format citations for the response model ───────────────────────────────
    formatted_citations = [
        Citation(
            source=c["source"],
            page=c["page"],
            snippet=c["snippet"],
        )
        for c in result["citations"]
    ]

    return AnswerResponse(
        question=result["question"],
        answer=result["answer"],
        citations=formatted_citations,
        time_taken_seconds=elapsed,
    )


@app.delete("/reset", tags=["Document"])
async def reset_system():
    """
    ENDPOINT: DELETE /reset
    PURPOSE:  Clear the vector store and app state. Start fresh.

    USE CASE: User wants to upload a new document and start over.
    """
    import glob

    # Clear the in-memory state
    app_state["vector_store"]    = None
    app_state["document_loaded"] = False
    app_state["current_doc"]     = None

    # Delete saved FAISS index files from disk
    vector_store_path = Path("vector_store")
    deleted_files = []
    for f in vector_store_path.glob(f"{STORE_NAME}*"):
        f.unlink()   # Delete the file
        deleted_files.append(str(f))

    logger.info(f"System reset. Deleted: {deleted_files}")

    return {
        "message": "System reset successfully. Upload a new document to begin.",
        "deleted_files": deleted_files,
    }


# ── Entry point: run with uvicorn ─────────────────────────────────────────────
# This block only runs when you execute: python main.py directly
# When using uvicorn from the command line, this block is skipped.
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))

    logger.info(f"Starting server at http://{host}:{port}")
    logger.info(f"API docs available at http://localhost:{port}/docs")

    uvicorn.run(
        "app.backend.main:app",  # "module:variable" format
        host=host,
        port=port,
        reload=False,            # Set True during development for auto-restart
        log_level="info",
    )
