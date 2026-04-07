"""
qa_chain.py
─────────────────────────────────────────────────────────────────────────────
WHAT THIS FILE DOES:
  1. Takes a user's question
  2. Retrieves the most relevant chunks from FAISS (semantic search)
  3. Builds a prompt: "Given this context... answer this question..."
  4. Passes the prompt to a free local LLM (flan-t5-base)
  5. Returns the answer AND the citations (which file, which page)

THINK OF IT LIKE:
  A research assistant who:
    - Finds the right pages in the book (retriever)
    - Reads those pages carefully (LLM reads context)
    - Writes you a clear answer with footnotes (answer + citations)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any

# Pipeline: Hugging Face utility to load a model + tokenizer in one step
from transformers import pipeline

# LangChain wrappers around Hugging Face models and prompts
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Our own modules from earlier phases
from app.backend.embeddings import (
    get_embedding_model,
    load_vector_store,
    ingest_and_store,
    vector_store_exists,
)
from app.backend.ingestion import ingest_document

from dotenv import load_dotenv

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# flan-t5-base: free, runs on CPU, no API key needed, great at Q&A
# "base" = ~250MB. There's also "large" (780MB) and "xl" (3GB) — base is fine for us.
LLM_MODEL      = os.getenv("LLM_MODEL", "google/flan-t5-base")
TOP_K_RESULTS  = int(os.getenv("TOP_K_RESULTS", 5))  # How many chunks to retrieve


# ── The Prompt Template ───────────────────────────────────────────────────────
# This is the EXACT text we send to the LLM.
# {context} and {question} are placeholders — LangChain fills them in automatically.
#
# WHY "Use ONLY the context below"?
#   We want the LLM to answer from the DOCUMENT, not from its training data.
#   This prevents hallucinations and ensures citations are accurate.
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based strictly on the provided document context.

Use ONLY the context below to answer the question.
If the answer is not in the context, say "I could not find the answer in the provided document."
Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""


def load_llm() -> HuggingFacePipeline:
    """
    WHAT IT DOES:
        Loads the flan-t5-base language model from Hugging Face.
        First run: downloads ~250MB (cached locally after that).

    HOW THE LLM WORKS:
        flan-t5-base is a "text-to-text" model:
        • Input:  "Answer this question given the context: ..."
        • Output: "AI helps with imaging by detecting tumors..."

        It was fine-tuned on thousands of Q&A tasks, so it's very good
        at reading a passage and extracting/summarizing an answer.

    RETURNS:
        HuggingFacePipeline: a LangChain-compatible LLM wrapper
    """

    logger.info(f"Loading LLM: {LLM_MODEL} (first run downloads ~250MB)...")

    # pipeline() is a Hugging Face high-level API.
    # task="text2text-generation" = give it text, get text back (perfect for Q&A)
    hf_pipeline = pipeline(
        task="text2text-generation",
        model=LLM_MODEL,          # Which model to load
        max_new_tokens=512,        # Max length of the answer it generates
        temperature=0.1,           # Low temperature = more focused, less random answers
                                   # (0 = deterministic, 1 = very creative/random)
        do_sample=True,            # Enable sampling (required when temperature > 0)
        repetition_penalty=1.3,    # Penalize repeating the same words (cleaner output)
        device=-1,                 # -1 = use CPU (0 = GPU if you had one)
    )

    # Wrap the Hugging Face pipeline so LangChain can use it
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    logger.info("LLM loaded successfully")
    return llm


def build_qa_chain(vector_store, llm: HuggingFacePipeline) -> RetrievalQA:
    """
    WHAT IT DOES:
        Assembles the full RAG pipeline by connecting:
          FAISS retriever  →  Prompt template  →  LLM

        This creates a "chain" — data flows through each step automatically.

    THE CHAIN FLOW:
        user_question
              ↓
        [Retriever] searches FAISS for top-k similar chunks
              ↓
        [PromptTemplate] fills {context} with chunks, {question} with the query
              ↓
        [LLM] reads the filled prompt and generates an answer
              ↓
        returns {"result": "...", "source_documents": [...]}

    PARAMETERS:
        vector_store:  FAISS vector store (from embeddings.py)
        llm:           The loaded language model (from load_llm())

    RETURNS:
        RetrievalQA: the assembled chain, ready to call with .invoke()
    """

    # Build the prompt from our template string above
    # input_variables tells LangChain which placeholders to fill
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    # Turn our FAISS vector store into a "retriever" object
    # search_type="similarity" = find chunks whose vectors are closest to the query
    # search_kwargs={"k": TOP_K_RESULTS} = return the top-5 most similar chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )

    # RetrievalQA is LangChain's built-in RAG chain
    # chain_type="stuff" = "stuff" all retrieved chunks into one prompt
    # (other options: "map_reduce" for very long docs, "refine" for iterative refinement)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,                       # The language model
        chain_type="stuff",            # How to combine chunks into the prompt
        retriever=retriever,           # The FAISS retriever
        return_source_documents=True,  # IMPORTANT: also return which chunks were used
        chain_type_kwargs={
            "prompt": prompt,          # Use our custom prompt template
            "verbose": False,          # Set True to see the full prompt being built
        },
    )

    logger.info("Q&A chain built successfully")
    return qa_chain


def format_citations(source_documents: list) -> List[Dict[str, Any]]:
    """
    WHAT IT DOES:
        Takes the raw source_documents returned by the chain
        and formats them into clean citation objects.

    BEFORE (raw):
        Document(page_content="AI helps...", metadata={"source": "file.pdf", "page": 0})

    AFTER (formatted):
        [{"source": "file.pdf", "page": 1, "snippet": "AI helps..."}]

    NOTE: page is stored as 0-indexed internally but we display as 1-indexed
          (so page 0 in the file = "Page 1" to the user)

    PARAMETERS:
        source_documents: list of Document objects from the QA chain

    RETURNS:
        list of dicts with clean citation info
    """

    citations = []
    seen = set()  # Track duplicates — same page can appear twice, show once

    for doc in source_documents:
        source = doc.metadata.get("source", "Unknown source")
        page   = doc.metadata.get("page", 0)
        snippet = doc.page_content[:200].strip()  # First 200 chars as a preview

        # Create a unique key to deduplicate (same source + page = same citation)
        key = f"{source}::page{page}"
        if key in seen:
            continue
        seen.add(key)

        # Extract just the filename (not the full path) for cleaner display
        filename = Path(source).name

        citations.append({
            "source":   filename,          # e.g. "sample_ai_healthcare.pdf"
            "page":     page + 1,          # Convert 0-indexed → 1-indexed for humans
            "full_path": source,           # Full path (used internally)
            "snippet":  snippet + "...",   # Preview of the text used
        })

    return citations


def answer_question(question: str, vector_store, llm: HuggingFacePipeline) -> Dict:
    """
    WHAT IT DOES:
        The MAIN function — takes a question, runs the full RAG pipeline,
        returns a structured response with answer + citations.

    PARAMETERS:
        question (str):  The user's question in plain English
        vector_store:    Loaded FAISS vector store
        llm:             Loaded language model

    RETURNS:
        dict: {
            "question":  the original question,
            "answer":    the generated answer,
            "citations": list of citation dicts (source, page, snippet)
        }
    """

    logger.info(f"Processing question: '{question}'")

    # Build the chain fresh each call (fast — no model reloading)
    qa_chain = build_qa_chain(vector_store, llm)

    # .invoke() runs the full pipeline:
    # question → retrieve chunks → fill prompt → LLM generates answer
    response = qa_chain.invoke({"query": question})

    # Extract the answer text from the response dict
    answer = response.get("result", "No answer generated.").strip()

    # Extract and format the citations
    source_docs = response.get("source_documents", [])
    citations   = format_citations(source_docs)

    logger.info(f"Answer generated. Citations found: {len(citations)}")

    return {
        "question":  question,
        "answer":    answer,
        "citations": citations,
    }


# ── Convenience: load everything needed for answering questions ───────────────
def load_rag_system(store_name: str = "main_index"):
    """
    WHAT IT DOES:
        Loads both the vector store and the LLM in one call.
        Used by the FastAPI backend to initialize the system at startup.

    RETURNS:
        (vector_store, llm): a tuple of both components
    """

    embedding_model = get_embedding_model()
    vector_store    = load_vector_store(embedding_model, store_name)
    llm             = load_llm()

    return vector_store, llm


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("\n" + "="*60)
    print("  RAG Q&A System — Full Pipeline Test")
    print("="*60)

    test_file = "data/uploads/sample_ai_healthcare.pdf"
    store_name = "main_index"

    # ── Step 1: Load or create the vector store ───────────────────────────────
    print("\n[1/3] Setting up vector store...")
    embedding_model = get_embedding_model()

    if vector_store_exists(store_name):
        print("  Found existing index — loading from disk (fast!)")
        vector_store = load_vector_store(embedding_model, store_name)
    else:
        print("  No index found — ingesting document and creating index...")
        chunks = ingest_document(test_file)
        vector_store = ingest_and_store(chunks, store_name)

    # ── Step 2: Load the LLM ──────────────────────────────────────────────────
    print("\n[2/3] Loading language model (downloads ~250MB on first run)...")
    llm = load_llm()

    # ── Step 3: Ask questions ─────────────────────────────────────────────────
    print("\n[3/3] Answering questions...\n")

    questions = [
        "How does AI help with medical imaging?",
        "What are the main challenges of AI in healthcare?",
        "What is federated learning?",
    ]

    for q in questions:
        print(f"Q: {q}")
        result = answer_question(q, vector_store, llm)

        print(f"A: {result['answer']}")
        print("Citations:")
        for c in result["citations"]:
            print(f"   - {c['source']}, Page {c['page']}")
            print(f"     Snippet: {c['snippet'][:100]}...")
        print("-" * 60)
