import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def generate_response(prompt, model="phi3:mini"):
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