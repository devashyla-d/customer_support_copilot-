from datetime import datetime
from pathlib import Path

import pandas as pd


LOG_FILE = Path(__file__).resolve().parent / "run_log.csv"
CSV_COLUMNS = ["timestamp", "question", "response", "model"]


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