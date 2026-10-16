from datetime import datetime
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from ingestion.document_manager import DOCUMENTS_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.csv"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNK_COLUMNS = [
    "chunk_id",
    "document_name",
    "chunk_number",
    "chunk_text",
    "created_timestamp",
]


def read_document_text(document_name: str) -> str:
    safe_name = Path(document_name).name
    document_path = DOCUMENTS_DIR / safe_name
    if not document_path.is_file():
        raise FileNotFoundError(f"Document '{safe_name}' was not found.")

    extension = document_path.suffix.lower()
    if extension in {".txt", ".md"}:
        document_text = document_path.read_text(encoding="utf-8")
    elif extension == ".pdf":
        try:
            document_text = "\n".join(
                page.extract_text() or "" for page in PdfReader(document_path).pages
            )
        except Exception as error:
            raise ValueError(f"Could not extract text from '{safe_name}': {error}") from error
    else:
        raise ValueError("Only PDF, TXT, and Markdown documents can be chunked.")

    if not document_text.strip():
        raise ValueError(f"Document '{safe_name}' is empty or contains no extractable text.")
    return document_text


def chunk_document(document_name: str, document_text: str) -> list[dict]:
    if not isinstance(document_text, str) or not document_text.strip():
        raise ValueError("Cannot generate chunks from an empty document.")

    created_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    chunks = []
    step_size = CHUNK_SIZE - CHUNK_OVERLAP

    for chunk_number, start in enumerate(range(0, len(document_text), step_size), start=1):
        chunk_text = document_text[start : start + CHUNK_SIZE]
        chunks.append(
            {
                "chunk_id": f"{document_name}::chunk-{chunk_number:04d}",
                "document_name": document_name,
                "chunk_number": chunk_number,
                "chunk_text": chunk_text,
                "created_timestamp": created_timestamp,
            }
        )

    return chunks


def _ensure_chunks_file() -> None:
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CHUNKS_PATH.exists():
        pd.DataFrame(columns=CHUNK_COLUMNS).to_csv(CHUNKS_PATH, index=False)


def get_saved_chunks(document_name: str | None = None) -> pd.DataFrame:
    _ensure_chunks_file()
    if CHUNKS_PATH.stat().st_size == 0:
        return pd.DataFrame(columns=CHUNK_COLUMNS)

    chunks = pd.read_csv(CHUNKS_PATH)
    if document_name is not None:
        chunks = chunks[chunks["document_name"] == document_name]
    return chunks


def save_chunks(chunks: list[dict]) -> None:
    if not chunks:
        raise ValueError("There are no chunks to save.")

    new_chunks = pd.DataFrame(chunks, columns=CHUNK_COLUMNS)
    document_names = new_chunks["document_name"].dropna().unique()
    if len(document_names) == 0:
        raise ValueError("Chunks must include a document name.")

    existing_chunks = get_saved_chunks()
    existing_chunks = existing_chunks[
        ~existing_chunks["document_name"].isin(document_names)
    ]
    all_chunks = pd.concat([existing_chunks, new_chunks], ignore_index=True)
    all_chunks.to_csv(CHUNKS_PATH, index=False)