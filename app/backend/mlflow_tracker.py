"""
mlflow_tracker.py  —  MLflow Experiment Tracking
─────────────────────────────────────────────────────────────────────────────
WHAT THIS FILE DOES:
  Wraps MLflow logging so every evaluation run is automatically recorded with:
    • Parameters  — what settings were used (model name, chunk size, etc.)
    • Metrics     — the scores (faithfulness, recall, etc.)
    • Artifacts   — the full JSON results file (for detailed inspection)

  Every run is saved to disk in mlflow_tracking/ and viewable in a
  beautiful web dashboard at http://localhost:5000

THINK OF IT LIKE:
  A lab notebook that writes itself.
  Scientist runs experiment → notebook records every detail automatically.
  Later, scientist opens notebook and compares all experiments side-by-side.

HOW TO VIEW THE DASHBOARD:
  mlflow ui --backend-store-uri mlflow_tracking --port 5000
  Then open: http://localhost:5000
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

import mlflow                       # The main MLflow library
import mlflow.artifacts             # For logging files as artifacts

from dotenv import load_dotenv
load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── MLflow Configuration ──────────────────────────────────────────────────────
# Where MLflow saves all run data (metrics, params, artifacts)
# This folder is created automatically if it doesn't exist
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlflow_tracking")

# The experiment name — groups related runs together
# Think of it like a folder name for a series of experiments
EXPERIMENT_NAME = "rag-doc-qa-evaluation"


def setup_mlflow():
    """
    WHAT IT DOES:
        Configures MLflow to save data locally and creates (or reuses)
        our experiment.

    EXPERIMENTS vs RUNS:
        Experiment = a research question, e.g. "Which embedding model is best?"
        Run        = one trial, e.g. "test with all-MiniLM-L6-v2, chunk_size=500"

        Many runs live inside one experiment.
        MLflow UI shows all runs as rows in a table for easy comparison.
    """

    # Tell MLflow to save data to our local folder (not a remote server)
    # In production you'd point this to a real MLflow server
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    # Create the experiment if it doesn't exist yet
    # If it already exists, this just returns the existing one
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow experiment: '{EXPERIMENT_NAME}'")


def log_evaluation_run(
    evaluation_results: dict,
    run_config: dict,
    results_file_path: str = None,
) -> str:
    """
    WHAT IT DOES:
        Logs a complete evaluation run to MLflow.
        Every call to this function creates one "run" in the experiment.

    WHAT GETS LOGGED:
        Parameters (settings used):
          • embedding_model  — which model converted text to vectors
          • llm_model        — which language model generated answers
          • chunk_size       — how large each text chunk was
          • chunk_overlap    — how much overlap between chunks
          • top_k            — how many chunks retrieved per question
          • num_questions    — how many questions were evaluated

        Metrics (scores achieved):
          • faithfulness     — hallucination score (higher = better)
          • answer_relevancy — on-topic score (higher = better)
          • context_recall   — retrieval quality (higher = better)
          • overall_score    — average of all 3

        Artifacts (files):
          • The full JSON results file (for detailed inspection later)

    PARAMETERS:
        evaluation_results (dict): Output from ragas_eval.run_evaluation()
        run_config (dict):         Settings used for this run
        results_file_path (str):   Path to the saved JSON results file

    RETURNS:
        str: The MLflow run ID (unique identifier for this run)
    """

    setup_mlflow()

    # mlflow.start_run() creates a new run and returns a context manager
    # Everything logged inside the "with" block belongs to this run
    with mlflow.start_run() as run:

        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")

        # ── Log Parameters ────────────────────────────────────────────────────
        # Parameters = the settings/config we used for this experiment
        # They help us understand WHY one run performed differently from another
        params = {
            "embedding_model": run_config.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "llm_model":       run_config.get("llm_model", "google/flan-t5-base"),
            "chunk_size":      run_config.get("chunk_size", 500),
            "chunk_overlap":   run_config.get("chunk_overlap", 50),
            "top_k":           run_config.get("top_k", 5),
            "num_questions":   run_config.get(
                "num_questions", len(evaluation_results.get("per_question", []))
            ),
            "document":        run_config.get("document", "unknown"),
            "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # mlflow.log_params() saves a dictionary of name→value pairs
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters")

        # ── Log Metrics ───────────────────────────────────────────────────────
        # Metrics = the actual scores/measurements from the evaluation
        # MLflow can plot these over time, compare across runs, etc.
        scores = evaluation_results.get("scores", {})

        metrics = {
            "faithfulness":     scores.get("faithfulness",     0.0),
            "answer_relevancy": scores.get("answer_relevancy", 0.0),
            "context_recall":   scores.get("context_recall",   0.0),
            "overall_score":    scores.get("overall",          0.0),
        }

        # mlflow.log_metrics() saves a dictionary of name→float pairs
        mlflow.log_metrics(metrics)
        logger.info(f"Logged metrics: {metrics}")

        # ── Log Per-Question Metrics ──────────────────────────────────────────
        # We also log each question's scores individually
        # This lets us see which questions the system struggles with
        per_question = evaluation_results.get("per_question", [])
        for i, q_result in enumerate(per_question):
            # Prefix "q{i}_" groups them by question number in the UI
            mlflow.log_metrics({
                f"q{i+1}_faithfulness":     q_result.get("faithfulness", 0.0),
                f"q{i+1}_answer_relevancy": q_result.get("answer_relevancy", 0.0),
                f"q{i+1}_context_recall":   q_result.get("context_recall", 0.0),
            })

        # ── Log Artifact (the full JSON report) ───────────────────────────────
        # Artifacts = files attached to a run (JSON, images, model files, etc.)
        # You can download them later from the MLflow UI
        if results_file_path and Path(results_file_path).exists():
            # mlflow.log_artifact() uploads the file to MLflow's artifact store
            mlflow.log_artifact(results_file_path, artifact_path="evaluation_reports")
            logger.info(f"Logged artifact: {results_file_path}")

        # ── Set Tags ─────────────────────────────────────────────────────────
        # Tags are like labels — free-form text for categorizing runs
        mlflow.set_tags({
            "project":    "rag-doc-qa-system",
            "stage":      "evaluation",
            "run_type":   "automated",
        })

        logger.info(f"MLflow run completed: {run_id}")
        logger.info(f"View at: http://localhost:5000 (after running mlflow ui)")

        return run_id


def get_best_run() -> dict | None:
    """
    WHAT IT DOES:
        Searches all past MLflow runs and returns the one with the
        highest overall_score metric.

    USE CASE:
        After trying different chunk sizes, models, etc., find which
        configuration gave the best results.

    RETURNS:
        dict with the best run's params and metrics, or None if no runs exist
    """

    setup_mlflow()

    # MlflowClient gives us programmatic access to runs
    client = mlflow.MlflowClient()

    # Get the experiment by name
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        logger.warning("No experiment found. Run an evaluation first.")
        return None

    # Search all runs in the experiment, ordered by overall_score descending
    # filter_string="" means "no filter — return all runs"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["metrics.overall_score DESC"],  # Best score first
        max_results=1,                             # Only the top 1
    )

    if not runs:
        logger.warning("No runs found in experiment.")
        return None

    best_run = runs[0]

    return {
        "run_id":   best_run.info.run_id,
        "params":   best_run.data.params,
        "metrics":  best_run.data.metrics,
        "start_time": datetime.fromtimestamp(
            best_run.info.start_time / 1000
        ).strftime("%Y-%m-%d %H:%M:%S"),
    }


def list_all_runs() -> list:
    """
    WHAT IT DOES:
        Returns a summary of all past evaluation runs, sorted by score.
        Useful for printing a comparison table in the terminal.

    RETURNS:
        list of dicts, each with run_id, params, and key metrics
    """

    setup_mlflow()
    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.overall_score DESC"],
    )

    summary = []
    for run in runs:
        summary.append({
            "run_id":           run.info.run_id[:8],   # First 8 chars is enough
            "timestamp":        run.data.params.get("timestamp", "?"),
            "llm_model":        run.data.params.get("llm_model", "?"),
            "chunk_size":       run.data.params.get("chunk_size", "?"),
            "faithfulness":     run.data.metrics.get("faithfulness", 0),
            "answer_relevancy": run.data.metrics.get("answer_relevancy", 0),
            "context_recall":   run.data.metrics.get("context_recall", 0),
            "overall_score":    run.data.metrics.get("overall_score", 0),
        })

    return summary


# ── Main: run a full evaluation and log it ────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from app.backend.ingestion import ingest_document
    from app.backend.embeddings import (
        get_embedding_model, ingest_and_store,
        load_vector_store, vector_store_exists,
    )
    from app.backend.qa_chain import load_llm
    from evaluation.ragas_eval import (
        EVAL_QUESTIONS, collect_rag_outputs,
        run_evaluation, save_results,
    )

    print("\n" + "=" * 62)
    print("  RAG System — MLflow Experiment Tracking")
    print("=" * 62)

    test_file  = "data/uploads/sample_ai_healthcare.pdf"
    store_name = "main_index"

    # ── Step 1: Set up RAG system ─────────────────────────────────────────────
    print("\n[1/5] Loading vector store...")
    embedding_model = get_embedding_model()
    if vector_store_exists(store_name):
        vector_store = load_vector_store(embedding_model, store_name)
    else:
        chunks = ingest_document(test_file)
        vector_store = ingest_and_store(chunks, store_name)

    print("[2/5] Loading LLM...")
    llm = load_llm()

    # ── Step 2: Run evaluation ────────────────────────────────────────────────
    print(f"\n[3/5] Running {len(EVAL_QUESTIONS)} evaluation questions...")
    collected   = collect_rag_outputs(EVAL_QUESTIONS, vector_store, llm)
    evaluation  = run_evaluation(collected)
    output_file = save_results(evaluation)

    scores = evaluation["scores"]
    print(f"\n  Scores computed:")
    print(f"    Faithfulness:     {scores['faithfulness']:.3f}")
    print(f"    Answer Relevancy: {scores['answer_relevancy']:.3f}")
    print(f"    Context Recall:   {scores['context_recall']:.3f}")
    print(f"    Overall:          {scores['overall']:.3f}")

    # ── Step 3: Log to MLflow ─────────────────────────────────────────────────
    print("\n[4/5] Logging to MLflow...")
    run_config = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model":       "google/flan-t5-base",
        "chunk_size":      500,
        "chunk_overlap":   50,
        "top_k":           5,
        "document":        "sample_ai_healthcare.pdf",
    }

    run_id = log_evaluation_run(evaluation, run_config, output_file)
    print(f"\n  Run logged! ID: {run_id}")

    # ── Step 4: Show all runs ─────────────────────────────────────────────────
    print("\n[5/5] All experiment runs so far:")
    all_runs = list_all_runs()

    if all_runs:
        print(f"\n  {'Run ID':<10} {'LLM':<20} {'Chunk':>6} {'Faith':>7} {'Relev':>7} {'Recall':>7} {'Overall':>8}")
        print("  " + "-" * 72)
        for r in all_runs:
            llm_short = str(r['llm_model']).split('/')[-1][:18]
            print(
                f"  {r['run_id']:<10} {llm_short:<20} "
                f"{r['chunk_size']:>6} "
                f"{r['faithfulness']:>7.3f} "
                f"{r['answer_relevancy']:>7.3f} "
                f"{r['context_recall']:>7.3f} "
                f"{r['overall_score']:>8.3f}"
            )

    print("\n" + "=" * 62)
    print("  To open the MLflow dashboard, run in Terminal:")
    print(f"  cd /Users/saitejanlp/Documents/rag-doc-qa-system")
    print(f"  source venv/bin/activate")
    print(f"  mlflow ui --backend-store-uri mlflow_tracking --port 5000")
    print("  Then open: http://localhost:5000")
    print("=" * 62 + "\n")
