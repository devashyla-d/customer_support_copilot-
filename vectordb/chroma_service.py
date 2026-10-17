import json
import math
from pathlib import Path

import chromadb
import pandas as pd

from embeddings.embedding_service import generate_embedding
from ingestion.chunker import CHUNKS_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.csv"
CHROMA_DB_PATH = PROJECT_ROOT / "data" / "vector_store" / "chroma_db"
COLLECTION_NAME = "enterprise_support_docs"
REQUIRED_COLUMNS = {
	"chunk_id",
	"document_name",
	"chunk_number",
	"embedding_vector",
}

_client = None
_client_path = None


def initialize_collection():
	global _client, _client_path

	if _client is None or _client_path != CHROMA_DB_PATH:
		CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
		_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
		_client_path = CHROMA_DB_PATH

	return _client.get_or_create_collection(
		name=COLLECTION_NAME,
		metadata={"hnsw:space": "cosine"},
	)


def _read_embedding_records() -> pd.DataFrame:
	if not EMBEDDINGS_PATH.is_file():
		raise FileNotFoundError("No embeddings file found. Generate embeddings first.")

	records = pd.read_csv(EMBEDDINGS_PATH, keep_default_na=False)
	missing_columns = REQUIRED_COLUMNS.difference(records.columns)
	if missing_columns:
		missing = ", ".join(sorted(missing_columns))
		raise ValueError(f"Embeddings file is missing required columns: {missing}.")

	if not CHUNKS_PATH.is_file():
		raise FileNotFoundError("No chunks file found. Generate document chunks first.")
	chunks = pd.read_csv(CHUNKS_PATH, keep_default_na=False)
	if not {"chunk_id", "chunk_text"}.issubset(chunks.columns):
		raise ValueError("Chunks file must include chunk_id and chunk_text columns.")

	chunk_texts = chunks[["chunk_id", "chunk_text"]].drop_duplicates(
		subset="chunk_id", keep="last"
	)
	records = records.merge(chunk_texts, on="chunk_id", how="left")
	records["chunk_text"] = records["chunk_text"].fillna("")
	return records


def index_embeddings() -> int:
	records = _read_embedding_records()
	records = records.drop_duplicates(subset="chunk_id", keep="last")

	ids = []
	documents = []
	embeddings = []
	metadatas = []

	for row in records.itertuples(index=False):
		chunk_id = str(row.chunk_id).strip()
		document_name = str(row.document_name).strip()
		chunk_text = str(getattr(row, "chunk_text", "")).strip()
		vector_text = str(row.embedding_vector).strip()

		if not chunk_id or not document_name or not chunk_text:
			continue
		if not vector_text:
			raise ValueError(f"Embedding vector is missing for chunk '{chunk_id}'.")

		try:
			vector = json.loads(vector_text)
			vector = [float(value) for value in vector]
			chunk_number = int(row.chunk_number)
		except (TypeError, ValueError, json.JSONDecodeError) as error:
			raise ValueError(f"Invalid embedding record for chunk '{chunk_id}'.") from error

		if not vector or not all(math.isfinite(value) for value in vector):
			raise ValueError(f"Invalid embedding vector for chunk '{chunk_id}'.")

		ids.append(chunk_id)
		documents.append(chunk_text)
		embeddings.append(vector)
		metadatas.append(
			{
				"document_name": document_name,
				"chunk_number": chunk_number,
			}
		)

	if not ids:
		raise ValueError("No valid embedding records are available to index.")

	collection = initialize_collection()
	collection.upsert(
		ids=ids,
		documents=documents,
		embeddings=embeddings,
		metadatas=metadatas,
	)

	indexed_ids = set(ids)
	existing_ids = set(collection.get(include=["metadatas"])["ids"])
	stale_ids = existing_ids - indexed_ids
	if stale_ids:
		collection.delete(ids=sorted(stale_ids))

	return len(ids)


def get_collection_inventory(limit: int = 100) -> tuple[int, list[dict]]:
	if limit < 1:
		raise ValueError("limit must be at least 1.")

	collection = initialize_collection()
	total_count = collection.count()
	if total_count == 0:
		return total_count, []

	result = collection.get(
		limit=min(limit, total_count),
		include=["documents", "metadatas", "embeddings"],
	)
	records = []
	for chunk_id, document, metadata, vector in zip(
		result["ids"],
		result["documents"],
		result["metadatas"],
		result["embeddings"],
	):
		if hasattr(vector, "tolist"):
			vector = vector.tolist()
		records.append(
			{
				"chunk_id": chunk_id,
				"document_name": metadata["document_name"],
				"chunk_number": metadata["chunk_number"],
				"chunk_text": document,
				"embedding_dimensions": len(vector),
				"embedding_vector": vector,
			}
		)
	return total_count, records


def search_documents(question: str, top_k: int = 5) -> list[dict]:
	if not isinstance(question, str) or not question.strip():
		raise ValueError("Enter a question to search.")
	if top_k < 1:
		raise ValueError("top_k must be at least 1.")

	collection = initialize_collection()
	collection_size = collection.count()
	if collection_size == 0:
		return []

	question_embedding = generate_embedding(question.strip())
	results = collection.query(
		query_embeddings=[question_embedding],
		n_results=min(top_k, collection_size),
		include=["documents", "metadatas", "distances"],
	)

	matches = []
	for chunk_id, document, metadata, distance in zip(
		results["ids"][0],
		results["documents"][0],
		results["metadatas"][0],
		results["distances"][0],
	):
		matches.append(
			{
				"chunk_id": chunk_id,
				"document_name": metadata["document_name"],
				"chunk_number": metadata["chunk_number"],
				"chunk_text": document,
				"similarity_score": 1.0 - float(distance),
			}
		)
	return matches
