from datetime import datetime
from pathlib import Path

import pandas as pd


LOG_FILE = Path(__file__).resolve().parent / "run_log.csv"
CSV_COLUMNS = ["timestamp", "question", "response", "model"]
RAG_LOG_FILE = Path(__file__).resolve().parent / "rag_run_log.csv"
RAG_CSV_COLUMNS = [
    "timestamp",
    "question",
    "retrieved_document_names",
    "retrieved_chunk_numbers",
    "response",
]


def log_interaction(question: str, response: str, model: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().astimezone().isoformat(),
                "question": question,
                "response": response,
                "model": model,
            }
        ],
        columns=CSV_COLUMNS,
    )
    write_header = not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0
    record.to_csv(LOG_FILE, mode="a", header=write_header, index=False)


def log_rag_interaction(question: str, sources: list[dict], response: str) -> None:
    RAG_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().astimezone().isoformat(),
                "question": question,
                "retrieved_document_names": "; ".join(
                    dict.fromkeys(source["document_name"] for source in sources)
                ),
                "retrieved_chunk_numbers": "; ".join(
                    str(source["chunk_number"]) for source in sources
                ),
                "response": response,
            }
        ],
        columns=RAG_CSV_COLUMNS,
    )
    write_header = not RAG_LOG_FILE.exists() or RAG_LOG_FILE.stat().st_size == 0
    record.to_csv(RAG_LOG_FILE, mode="a", header=write_header, index=False)