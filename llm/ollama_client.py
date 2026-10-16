import json

import requests

from config import MODEL_NAME

OLLAMA_URL = "http://localhost:11434/api/generate"


def generate_response(prompt, model=MODEL_NAME):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 128},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def generate_response_stream(prompt, model=MODEL_NAME):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": 128},
    }

    with requests.post(
        OLLAMA_URL,
        json=payload,
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line:
                continue

            chunk = json.loads(line)
            text = chunk.get("response", "")
            if text:
                yield text

            if chunk.get("done"):
                break