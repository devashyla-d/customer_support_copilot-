from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import truststore

from ingestion.chunker import CHUNKS_PATH

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.csv"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_COLUMNS = [
    "chunk_id",
    "document_name",
    "chunk_number",
    "embedding_vector",
    "created_timestamp",
]

_model = None
_system_trust_initialized = False


def _get_model() -> SentenceTransformer:
    global _model, _system_trust_initialized
    if _model is None:
        if not _system_trust_initialized:
            truststore.inject_into_ssl()
            _system_trust_initialized = True

        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def generate_embedding(text: str) -> list[float]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Cannot generate an embedding for empty text.")
    return _get_model().encode(text).tolist()


def _read_chunks() -> pd.DataFrame:
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError("No chunks file found. Generate document chunks first.")

    chunks = pd.read_csv(CHUNKS_PATH, keep_default_na=False)
    required_columns = {"chunk_id", "document_name", "chunk_number", "chunk_text"}
    missing_columns = required_columns.difference(chunks.columns)
    if missing_columns:
        raise ValueError(f"Chunks file is missing columns: {', '.join(sorted(missing_columns))}.")
    return chunks


def _build_embedding_records(chunks: pd.DataFrame) -> list[dict]:
    records = []
    created_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    for row in chunks.drop_duplicates(subset="chunk_id", keep="last").itertuples(index=False):
        text = str(row.chunk_text)
        if not text.strip():
            continue

        records.append(
            {
                "chunk_id": row.chunk_id,
                "document_name": row.document_name,
                "chunk_number": row.chunk_number,
                "embedding_vector": json.dumps(generate_embedding(text)),
                "created_timestamp": created_timestamp,
            }
        )
    return records


def _ensure_embeddings_file() -> None:
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not EMBEDDINGS_PATH.exists():
        pd.DataFrame(columns=EMBEDDING_COLUMNS).to_csv(EMBEDDINGS_PATH, index=False)


def _read_existing_embeddings() -> pd.DataFrame:
    _ensure_embeddings_file()
    if EMBEDDINGS_PATH.stat().st_size == 0:
        return pd.DataFrame(columns=EMBEDDING_COLUMNS)
    return pd.read_csv(EMBEDDINGS_PATH, keep_default_na=False)


def _replace_embeddings(records: list[dict], document_names: set[str] | None = None) -> None:
    new_embeddings = pd.DataFrame(records, columns=EMBEDDING_COLUMNS)
    if document_names is None:
        combined = new_embeddings
    else:
        existing = _read_existing_embeddings()
        existing = existing[~existing["document_name"].isin(document_names)]
        combined = pd.concat([existing, new_embeddings], ignore_index=True)

    _ensure_embeddings_file()
    combined.to_csv(EMBEDDINGS_PATH, index=False)


def get_embedding_inventory(document_name: str | None = None) -> pd.DataFrame:
    embeddings = _read_existing_embeddings()
    if document_name is not None:
        embeddings = embeddings[embeddings["document_name"] == document_name]
    return embeddings


def generate_embeddings_for_document(document_name: str) -> list[dict]:
    chunks = _read_chunks()
    document_chunks = chunks[chunks["document_name"] == document_name]
    if document_chunks.empty:
        raise ValueError(f"No chunks found for '{document_name}'. Generate chunks first.")

    records = _build_embedding_records(document_chunks)
    if not records:
        raise ValueError(f"'{document_name}' has no non-empty chunks to embed.")

    _replace_embeddings(records, {document_name})
    return records


def generate_embeddings_for_all_documents() -> list[dict]:
    chunks = _read_chunks()
    non_empty_chunks = chunks[chunks["chunk_text"].astype(str).str.strip().ne("")]
    if non_empty_chunks.empty:
        raise ValueError("No non-empty chunks are available to embed.")

    records = _build_embedding_records(non_empty_chunks)
    _replace_embeddings(records)
    return records