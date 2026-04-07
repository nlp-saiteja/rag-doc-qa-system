"""
ingestion.py
─────────────────────────────────────────────────────────────────────────────
WHAT THIS FILE DOES:
  1. Takes a PDF (or text) file from the user
  2. Reads all the text out of it
  3. Splits that text into small overlapping chunks
  4. Returns those chunks — ready to be converted into vectors (embeddings)

THINK OF IT LIKE:
  A librarian who reads your book, then creates a stack of index cards,
  each card containing a small passage from the book.
─────────────────────────────────────────────────────────────────────────────
"""

import os                          # Built-in: for working with file paths
import logging                     # Built-in: for printing info/warning messages
from pathlib import Path           # Built-in: a cleaner way to handle file paths

# LangChain document loaders — these know how to READ different file types
from langchain_community.document_loaders import PyPDFLoader    # Reads PDF files
from langchain_community.document_loaders import TextLoader     # Reads plain .txt files
from langchain_community.document_loaders import Docx2txtLoader # Reads .docx Word files

# LangChain text splitter — this CUTS the text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# python-dotenv: loads our .env file so we can use CHUNK_SIZE etc.
from dotenv import load_dotenv

# ── Load environment variables from .env file ─────────────────────────────────
# This reads .env and makes CHUNK_SIZE, CHUNK_OVERLAP etc. available via os.getenv()
load_dotenv()

# ── Set up logging ────────────────────────────────────────────────────────────
# Instead of print(), we use logging — it adds timestamps and log levels (INFO, ERROR)
# Example output: 2024-01-15 10:23:45 - ingestion - INFO - Loading document...
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)  # __name__ = "ingestion" (the module name)


# ── Constants from .env (with sensible defaults if .env is missing) ───────────
# int() converts the string from .env into a number
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 500))    # Max characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))  # Characters shared between chunks


def load_document(file_path: str) -> list:
    """
    WHAT IT DOES:
        Reads a document file and returns a list of LangChain "Document" objects.
        Each Document has two parts:
          - .page_content  → the actual text from that page
          - .metadata      → info like {"source": "file.pdf", "page": 3}

    SUPPORTED FILE TYPES:
        .pdf, .txt, .docx

    PARAMETERS:
        file_path (str): Full path to the file, e.g. "/Users/you/docs/report.pdf"

    RETURNS:
        list[Document]: A list of Document objects (one per page for PDFs)
    """

    # Convert the string path to a Path object for easier manipulation
    path = Path(file_path)

    # Check if the file actually exists — fail early with a clear error
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get the file extension in lowercase, e.g. ".PDF" → ".pdf"
    extension = path.suffix.lower()

    logger.info(f"Loading document: {path.name} (type: {extension})")

    # ── Choose the right loader based on file type ────────────────────────────
    if extension == ".pdf":
        # PyPDFLoader reads each PAGE as a separate Document object
        loader = PyPDFLoader(file_path)

    elif extension == ".txt":
        # TextLoader reads the whole file as ONE Document object
        loader = TextLoader(file_path, encoding="utf-8")

    elif extension == ".docx":
        # Docx2txtLoader reads Word documents
        loader = Docx2txtLoader(file_path)

    else:
        # We don't support this file type yet — raise a clear error
        raise ValueError(
            f"Unsupported file type: '{extension}'. "
            f"Supported types: .pdf, .txt, .docx"
        )

    # .load() actually reads the file and returns the list of Document objects
    documents = loader.load()

    logger.info(f"Loaded {len(documents)} page(s) from '{path.name}'")

    return documents


def split_documents(documents: list) -> list:
    """
    WHAT IT DOES:
        Takes the list of Document objects from load_document()
        and splits them into smaller overlapping chunks.

    WHY WE NEED THIS:
        AI models have a "context window" limit — they can't process
        huge documents at once. By splitting into ~500-character chunks,
        we can search only the RELEVANT pieces (not the whole book).

    HOW OVERLAP WORKS:
        chunk_size=500, chunk_overlap=50 means:
        ┌─────────── Chunk 1 (500 chars) ────────────┐
                                   ┌── Chunk 2 (500 chars) ──────────────────┐
                                   ↑
                              50-char overlap
        This ensures sentences at boundaries aren't cut in half.

    PARAMETERS:
        documents (list): Output from load_document()

    RETURNS:
        list[Document]: Many smaller Document chunks, each with metadata
                        telling us which source file and page they came from
    """

    # RecursiveCharacterTextSplitter is the best general-purpose splitter.
    # It tries to split on: paragraphs → sentences → words → characters
    # (in that order), so it avoids cutting mid-sentence whenever possible.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,       # Max characters in each chunk (from .env)
        chunk_overlap=CHUNK_OVERLAP, # Characters to repeat between chunks (from .env)
        length_function=len,         # How to measure length (count characters)
        add_start_index=True,        # Adds "start_index" to metadata (useful for citations)
    )

    # split_documents() takes our list of Documents and returns MORE (smaller) Documents
    chunks = text_splitter.split_documents(documents)

    logger.info(
        f"Split into {len(chunks)} chunks "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )

    return chunks


def ingest_document(file_path: str) -> list:
    """
    WHAT IT DOES:
        The MAIN function of this file — combines load + split in one call.
        This is what the rest of our app will use.

    PIPELINE:
        file_path  →  load_document()  →  split_documents()  →  chunks

    PARAMETERS:
        file_path (str): Path to the document to ingest

    RETURNS:
        list[Document]: Ready-to-embed chunks with metadata
    """

    logger.info(f"Starting ingestion for: {file_path}")

    # Step 1: Load the raw document
    documents = load_document(file_path)

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    # Log a sample chunk so we can see what's inside (helpful for debugging)
    if chunks:
        logger.info(f"Sample chunk preview:\n---\n{chunks[0].page_content[:200]}...\n---")
        logger.info(f"Sample chunk metadata: {chunks[0].metadata}")

    logger.info(f"Ingestion complete. Total chunks ready: {len(chunks)}")

    return chunks


# ── Quick test — only runs when you execute this file directly ────────────────
# In Python: "if __name__ == '__main__'" means "only run this block if I'm
# running THIS file directly (not importing it from another file)"
if __name__ == "__main__":
    import sys

    # Allow running as: python ingestion.py path/to/file.pdf
    if len(sys.argv) < 2:
        print("Usage: python ingestion.py <path_to_document>")
        print("Example: python ingestion.py data/uploads/sample.pdf")
        sys.exit(1)

    test_file = sys.argv[1]
    chunks = ingest_document(test_file)

    print(f"\nResults:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Chunk 1 text (first 300 chars):\n  {chunks[0].page_content[:300]}")
    print(f"  Chunk 1 metadata: {chunks[0].metadata}")
