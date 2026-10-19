from pathlib import Path

from config import MODEL_NAME
from llm.ollama_client import generate_response
from vectordb.chroma_service import search_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.txt"
MIN_SIMILARITY_SCORE = 0.35
NO_EVIDENCE = "NO_EVIDENCE"

RAG_INSTRUCTIONS = """Use only the information provided in the retrieved context below.
If the answer cannot be determined from the context, respond with:
NO_EVIDENCE
Do not use external knowledge."""


def _build_context(results: list[dict]) -> str:
    context_blocks = []
    for source_number, result in enumerate(results, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"Source {source_number}",
                    f"Document Name: {result['document_name']}",
                    f"Chunk Number: {result['chunk_number']}",
                    f"Similarity Score: {result['similarity_score']:.4f}",
                    f"Content: {result['chunk_text']}",
                ]
            )
        )
    return "\n\n".join(context_blocks)


def _build_prompt(question: str, context: str) -> str:
    system_prompt = (
        SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
        if SYSTEM_PROMPT_PATH.is_file()
        else "You are a customer support assistant."
    )
    return f"""{system_prompt}

{RAG_INSTRUCTIONS}

Retrieved Context:
{context}

Question:
{question}

Answer:"""


def answer_question(question: str, top_k: int = 5) -> dict:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Enter a question to ask.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    retrieved = search_documents(question.strip(), top_k=top_k)
    sources = [
        result
        for result in retrieved
        if result["similarity_score"] >= MIN_SIMILARITY_SCORE
    ]

    if not sources:
        return {"answer": NO_EVIDENCE, "sources": [], "debug_prompt": None}

    context = _build_context(sources)
    prompt = _build_prompt(question.strip(), context)
    answer = generate_response(prompt, model=MODEL_NAME).strip()
    if answer.splitlines() and answer.splitlines()[0].strip() == NO_EVIDENCE:
        answer = NO_EVIDENCE

    return {
        "answer": answer or NO_EVIDENCE,
        "sources": sources,
        "debug_prompt": prompt,
    }