"""
embeddings.py
─────────────────────────────────────────────────────────────────────────────
WHAT THIS FILE DOES:
  1. Loads a free Hugging Face embedding model (runs on YOUR machine, no API cost)
  2. Converts document chunks (text) → vectors (lists of numbers)
  3. Stores those vectors in a FAISS index (a super-fast search database)
  4. Saves the FAISS index to disk so we don't re-compute it every time
  5. Loads the FAISS index back from disk when needed

THINK OF IT LIKE:
  A translator that converts every index card (chunk) into a GPS coordinate.
  Then FAISS is a map — given your question's GPS coordinate, it finds
  the nearest index cards instantly.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
from pathlib import Path

# HuggingFaceEmbeddings: wraps a sentence-transformers model from Hugging Face
# It gives us a simple .embed_documents() and .embed_query() interface
from langchain_huggingface import HuggingFaceEmbeddings

# FAISS: the vector database — stores vectors and finds nearest neighbors fast
from langchain_community.vectorstores import FAISS

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
# The embedding model we use — "all-MiniLM-L6-v2" is:
#   • Free and open-source (no API key needed at runtime)
#   • Fast (runs in seconds on CPU)
#   • Small (only 80MB download)
#   • Produces 384-dimensional vectors
#   • Excellent quality for semantic search tasks
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Where we save the FAISS index files on disk
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store")


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    WHAT IT DOES:
        Loads the Hugging Face embedding model.
        The first time you run this, it downloads the model (~80MB).
        After that it's cached locally — no re-download needed.

    HOW EMBEDDINGS WORK:
        The model is a neural network trained on millions of sentence pairs.
        It learned that "dog" and "puppy" should be close together,
        and "dog" and "airplane" should be far apart — in vector space.

    RETURNS:
        HuggingFaceEmbeddings: a LangChain embedding model object
    """

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")

    # model_kwargs: settings passed to the underlying PyTorch model
    # device="cpu" means run on CPU (not GPU) — works on all Macs
    model_kwargs = {"device": "cpu"}

    # encode_kwargs: settings for how vectors are computed
    # normalize_embeddings=True makes all vectors the same length (unit vectors)
    # This improves similarity search accuracy
    encode_kwargs = {"normalize_embeddings": True}

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    logger.info("Embedding model loaded successfully")
    return embedding_model


def create_vector_store(chunks: list, embedding_model: HuggingFaceEmbeddings) -> FAISS:
    """
    WHAT IT DOES:
        Takes document chunks → converts each to a vector → stores in FAISS.

    STEP BY STEP:
        chunks = ["AI helps doctors...", "Radiology uses ML...", ...]
              ↓  (embedding model converts each to a vector)
        vectors = [[0.23, -0.87, ...], [0.11, 0.94, ...], ...]
              ↓  (FAISS stores all vectors in an optimized index)
        FAISS index = ready for lightning-fast similarity search

    PARAMETERS:
        chunks (list):                  Output from ingestion.split_documents()
        embedding_model:                Output from get_embedding_model()

    RETURNS:
        FAISS: a vector store object with all chunks indexed
    """

    logger.info(f"Creating FAISS vector store from {len(chunks)} chunks...")

    # FAISS.from_documents() does two things in one call:
    #   1. Embeds every chunk (calls embedding_model on each piece of text)
    #   2. Builds the FAISS index from those vectors
    # This may take 10-30 seconds depending on how many chunks you have
    vector_store = FAISS.from_documents(
        documents=chunks,           # Our text chunks with metadata
        embedding=embedding_model,  # The model that converts text → vectors
    )

    logger.info("FAISS vector store created successfully")
    return vector_store


def save_vector_store(vector_store: FAISS, store_name: str = "main_index") -> str:
    """
    WHAT IT DOES:
        Saves the FAISS index to disk as two files:
          • vector_store/main_index.faiss  → the actual vector data
          • vector_store/main_index.pkl    → the text chunks + metadata

        WHY SAVE? So we don't re-embed documents every time we restart.
        Embedding takes time. Once saved, loading takes < 1 second.

    PARAMETERS:
        vector_store (FAISS): The vector store to save
        store_name (str):     Name for the saved files (default: "main_index")

    RETURNS:
        str: The path where files were saved
    """

    # Make sure the vector_store/ folder exists
    save_path = Path(VECTOR_STORE_PATH)
    save_path.mkdir(parents=True, exist_ok=True)

    # save_local() saves both .faiss and .pkl files
    full_path = str(save_path / store_name)
    vector_store.save_local(full_path)

    logger.info(f"Vector store saved to: {full_path}")
    return full_path


def load_vector_store(
    embedding_model: HuggingFaceEmbeddings,
    store_name: str = "main_index"
) -> FAISS:
    """
    WHAT IT DOES:
        Loads a previously saved FAISS index from disk back into memory.
        This is fast (< 1 second) because we're just reading files,
        not re-computing embeddings.

    PARAMETERS:
        embedding_model:  Needed to embed new queries at search time
        store_name (str): Name of the saved index to load

    RETURNS:
        FAISS: the loaded vector store, ready to search

    RAISES:
        FileNotFoundError: if no saved index exists yet
    """

    load_path = str(Path(VECTOR_STORE_PATH) / store_name)

    # Check if the index files actually exist on disk
    if not Path(f"{load_path}.faiss").exists():
        raise FileNotFoundError(
            f"No vector store found at '{load_path}'.\n"
            f"Please ingest a document first to create the index."
        )

    logger.info(f"Loading vector store from: {load_path}")

    # allow_dangerous_deserialization=True is required by LangChain for FAISS
    # It's safe here because WE created these files ourselves
    vector_store = FAISS.load_local(
        load_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )

    logger.info("Vector store loaded successfully")
    return vector_store


def vector_store_exists(store_name: str = "main_index") -> bool:
    """
    WHAT IT DOES:
        Quick check — does a saved FAISS index exist on disk?
        Used by other parts of the app to decide whether to load or create.

    RETURNS:
        bool: True if index exists, False if we need to create one
    """
    faiss_file = Path(VECTOR_STORE_PATH) / f"{store_name}.faiss"
    return faiss_file.exists()


def ingest_and_store(chunks: list, store_name: str = "main_index") -> FAISS:
    """
    WHAT IT DOES:
        The MAIN convenience function — combines everything:
          1. Load embedding model
          2. Create FAISS index from chunks
          3. Save to disk
          4. Return the vector store ready for searching

    PARAMETERS:
        chunks (list):    Chunks from ingestion.ingest_document()
        store_name (str): Name to save the index under

    RETURNS:
        FAISS: ready-to-search vector store
    """

    # Step 1: Load the embedding model
    embedding_model = get_embedding_model()

    # Step 2: Convert chunks to vectors + build FAISS index
    vector_store = create_vector_store(chunks, embedding_model)

    # Step 3: Save the index to disk for next time
    save_vector_store(vector_store, store_name)

    return vector_store


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Import the ingestion module we built in Phase 3
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))  # project root
    from app.backend.ingestion import ingest_document

    test_file = "data/uploads/sample_ai_healthcare.pdf"

    print("\n=== STEP 1: Ingest the document ===")
    chunks = ingest_document(test_file)
    print(f"Got {len(chunks)} chunks from the PDF")

    print("\n=== STEP 2: Create + Save FAISS Vector Store ===")
    vector_store = ingest_and_store(chunks)

    print("\n=== STEP 3: Test Semantic Search ===")
    query = "How does AI help with medical imaging?"
    print(f"Query: '{query}'")

    # similarity_search() converts the query to a vector, then finds the
    # top-k most similar chunk-vectors in our FAISS index
    results = vector_store.similarity_search(query, k=2)

    print(f"\nTop {len(results)} most relevant chunks found:\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}, "
              f"Page: {doc.metadata.get('page', '?')}")
        print(f"Text: {doc.page_content[:300]}")
        print()
