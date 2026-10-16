from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
REGISTRY_PATH = PROJECT_ROOT / "data" / "structured" / "document_registry.csv"
REGISTRY_COLUMNS = ["document_name", "file_type", "upload_timestamp", "status"]
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _ensure_registry() -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        pd.DataFrame(columns=REGISTRY_COLUMNS).to_csv(REGISTRY_PATH, index=False)


def save_document(uploaded_file):
    document_name = Path(uploaded_file.name).name
    extension = Path(document_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Upload a PDF, TXT, or Markdown file.")

    file_content = uploaded_file.getvalue()
    if not file_content:
        raise ValueError("Empty files cannot be uploaded.")

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = DOCUMENTS_DIR / document_name
    if destination.exists():
        raise ValueError(f"A document named '{document_name}' already exists.")

    _ensure_registry()
    record = {
        "document_name": document_name,
        "file_type": extension.lstrip("."),
        "upload_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "Active",
    }

    destination.write_bytes(file_content)
    pd.DataFrame([record], columns=REGISTRY_COLUMNS).to_csv(
        REGISTRY_PATH,
        mode="a",
        header=REGISTRY_PATH.stat().st_size == 0,
        index=False,
    )
    return record


def get_document_inventory() -> pd.DataFrame:
    _ensure_registry()
    return pd.read_csv(REGISTRY_PATH)