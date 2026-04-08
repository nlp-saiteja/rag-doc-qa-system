# RAG-Powered Intelligent Document Q&A System

A production-grade **Retrieval-Augmented Generation (RAG)** system that lets you upload any document and ask questions about it in plain English — with cited answers telling you exactly which page the answer came from.

Built entirely with **free, open-source tools** that run locally on your machine. No OpenAI API key needed.

---

## What It Does

```
You upload a PDF  →  Ask a question  →  Get a cited answer in seconds

"What are the challenges of AI in healthcare?"
→ "Data privacy regulations like HIPAA and GDPR restrict how patient
   data can be used for training models..."
   📎 Source: report.pdf — Page 3
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                           │
│                    http://localhost:8501                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │  HTTP requests
┌─────────────────────▼───────────────────────────────────────────┐
│                  Streamlit Frontend                              │
│         Upload UI · Question Input · Answer Display             │
└─────────────────────┬───────────────────────────────────────────┘
                      │  REST API calls
┌─────────────────────▼───────────────────────────────────────────┐
│                   FastAPI Backend  :8000                         │
│    /upload  /ask  /status  /health  /reset                       │
└──────┬──────────────┬──────────────────────────────────────────┘
       │              │
┌──────▼──────┐  ┌────▼──────────────────────────────────────────┐
│  Ingestion  │  │              Q&A Pipeline                       │
│             │  │                                                 │
│ PDF → Chunks│  │  Question → Embedding → FAISS Search           │
│ (LangChain) │  │         → Top-K Chunks → Prompt → LLM         │
└──────┬──────┘  │         → Answer + Citations                   │
       │         └────┬──────────────────────────────────────────┘
┌──────▼──────┐  ┌────▼──────────┐  ┌──────────────────────────┐
│  HuggingFace│  │     FAISS     │  │      flan-t5-base        │
│  Embeddings │  │  Vector Store │  │    (Local LLM, free)     │
│ all-MiniLM  │  │  (on disk)    │  │   No API key needed      │
└─────────────┘  └───────────────┘  └──────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│               Supporting Services                                │
│   MLflow :5000  (experiment tracking & model lifecycle)         │
│   RAGAS         (retrieval accuracy evaluation — offline)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| **RAG Framework** | LangChain 0.2 | Chains ingestion → retrieval → generation |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Converts text to vectors (free, local, 80MB) |
| **Vector Database** | FAISS (Meta) | Sub-millisecond similarity search at scale |
| **Language Model** | `google/flan-t5-base` | Generates answers from retrieved context |
| **Backend API** | FastAPI + Uvicorn | REST endpoints for upload, Q&A, health |
| **Frontend UI** | Streamlit | Interactive web interface |
| **Evaluation** | RAGAS (embedding-based) | Measures faithfulness, relevancy, recall |
| **Experiment Tracking** | MLflow | Logs every eval run; compare configs |
| **Containerization** | Docker + Docker Compose | One-command deployment across environments |

---

## Project Structure

```
rag-doc-qa-system/
│
├── app/
│   ├── backend/
│   │   ├── ingestion.py        # PDF/TXT/DOCX loader + text chunker
│   │   ├── embeddings.py       # HuggingFace embeddings + FAISS index
│   │   ├── qa_chain.py         # RAG pipeline: retrieve → prompt → answer
│   │   ├── main.py             # FastAPI server with 5 REST endpoints
│   │   └── mlflow_tracker.py   # MLflow logging for every eval run
│   │
│   └── frontend/
│       └── streamlit_app.py    # Full web UI with upload + Q&A history
│
├── evaluation/
│   ├── ragas_eval.py           # 3-metric evaluation pipeline
│   └── results/                # Timestamped JSON reports per run
│
├── data/
│   └── uploads/                # Uploaded documents stored here
│
├── vector_store/               # FAISS index persisted to disk
├── mlflow_tracking/            # MLflow experiment database
├── logs/                       # Backend and frontend logs
│
├── Dockerfile                  # Container recipe for backend + frontend
├── docker-compose.yml          # Orchestrates 3 services together
├── requirements.txt            # All pinned Python dependencies
└── .env                        # API tokens and config (gitignored)
```

---

## How RAG Works — Plain English

Traditional search finds documents with **matching keywords**.
RAG finds documents with **matching meaning** — then *reads* them to write you an answer.

```
Step 1 — Ingest
  Your PDF (100 pages)
    → LangChain splits into 500-char overlapping chunks
    → Each chunk → 384-dimensional vector (HuggingFace embedding)
    → All vectors stored in FAISS index on disk

Step 2 — Retrieve
  Your question → also converted to a vector
    → FAISS finds the top-5 most semantically similar chunks
    → (not keyword matching — actual meaning similarity)

Step 3 — Generate
  Retrieved chunks + your question → filled into a prompt template
    → flan-t5-base reads the prompt and writes an answer
    → Citations extracted from chunk metadata (source file + page)

Step 4 — Return
  { answer, citations: [{source, page, snippet}], time_taken_seconds }
```

---

## Getting Started

### Option A — Run Locally (No Docker)

**Prerequisites:** Python 3.12, a free [HuggingFace account](https://huggingface.co/settings/tokens)

```bash
# 1. Clone the repository
git clone https://github.com/nlp-saiteja/rag-doc-qa-system.git
cd rag-doc-qa-system

# 2. Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install --upgrade pip setuptools
pip install -r requirements.txt

# 4. Add your HuggingFace token to .env
#    Get it free at: https://huggingface.co/settings/tokens
cp .env.example .env              # then edit .env

# 5. Start the backend (Terminal 1)
PYTHONPATH=. uvicorn app.backend.main:app --port 8000

# 6. Start the frontend (Terminal 2)
streamlit run app/frontend/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

---

### Option B — Run with Docker (Recommended)

**Prerequisites:** [OrbStack](https://orbstack.dev) or [Docker Desktop](https://www.docker.com/products/docker-desktop/)

```bash
# 1. Clone the repository
git clone https://github.com/nlp-saiteja/rag-doc-qa-system.git
cd rag-doc-qa-system

# 2. Add your HuggingFace token
cp .env.example .env              # then edit .env

# 3. Build and start all services
docker compose up -d

# 4. Wait ~60 seconds for models to load, then open:
#    Streamlit UI  →  http://localhost:8501
#    FastAPI Docs  →  http://localhost:8000/docs
#    MLflow UI     →  http://localhost:5000
```

```bash
# Stop everything
docker compose down

# View backend logs
docker compose logs -f backend

# Rebuild after code changes
docker compose up -d --build
```

---

## API Endpoints

The FastAPI backend exposes a clean REST API — fully documented at `http://localhost:8000/docs`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check server is running |
| `GET` | `/status` | Check if document + models are loaded |
| `POST` | `/upload` | Upload a PDF/TXT/DOCX file |
| `POST` | `/ask` | Ask a question, get answer + citations |
| `DELETE` | `/reset` | Clear vector store, start fresh |

**Example — Ask a question:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main challenges discussed?"}'
```

**Response:**
```json
{
  "question": "What are the main challenges discussed?",
  "answer": "The main challenges include data privacy regulations...",
  "citations": [
    {
      "source": "report.pdf",
      "page": 3,
      "snippet": "Data privacy regulations like HIPAA..."
    }
  ],
  "time_taken_seconds": 1.97
}
```

---

## Evaluation

The RAGAS evaluation framework measures system quality across 3 metrics using embedding-based semantic similarity — fully offline, no OpenAI key needed.

```bash
# Run evaluation
PYTHONPATH=. python evaluation/ragas_eval.py
```

| Metric | What it measures |
|---|---|
| **Faithfulness** | Does the answer only use info from the document? (hallucination check) |
| **Answer Relevancy** | Does the answer actually address the question? |
| **Context Recall** | Did FAISS retrieve the right chunks? (retrieval quality) |

Results are saved as timestamped JSON reports in `evaluation/results/`.

---

## Experiment Tracking with MLflow

Every evaluation run is automatically logged to MLflow — parameters, metrics, and the full JSON report as an artifact.

```bash
# Run evaluation + log to MLflow
PYTHONPATH=. python app/backend/mlflow_tracker.py

# Open the MLflow dashboard
mlflow ui --backend-store-uri mlflow_tracking --port 5000
# Then open: http://localhost:5000
```

The dashboard shows all runs in a comparison table — making it easy to see which model, chunk size, or config produced the best results.

---

## Supported File Types

| Format | Extension |
|---|---|
| PDF documents | `.pdf` |
| Plain text | `.txt` |
| Word documents | `.docx` |

---

## Environment Variables

Create a `.env` file (copy from `.env.example`):

```env
# Required
HUGGINGFACE_API_TOKEN=hf_your_token_here

# Optional (defaults shown)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=google/flan-t5-base
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
API_HOST=0.0.0.0
API_PORT=8000
MLFLOW_TRACKING_URI=./mlflow_tracking
```

---

## What I Learned Building This

- **How RAG works end-to-end** — from raw PDF bytes to a cited natural language answer
- **Semantic search vs keyword search** — why vector similarity finds better results than CTRL+F
- **LangChain pipelines** — chaining loaders, splitters, retrievers, and LLMs
- **FastAPI design patterns** — lifespan events, Pydantic models, file uploads, CORS
- **Docker fundamentals** — images, containers, volumes, multi-service compose
- **ML evaluation** — why you need metrics like RAGAS, not just "does it look right?"
- **MLflow tracking** — reproducible experiments and model lifecycle management

---

## License

MIT License — free to use, modify, and distribute.

---

*Built from scratch as a portfolio project demonstrating production-grade RAG system design.*
