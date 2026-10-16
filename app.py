from pathlib import Path

import streamlit as st
import requests

from llm.ollama_client import generate_response
from logs.logger import log_interaction

MODEL_NAME = "phi3:mini"

st.set_page_config(
    page_title="Customer Support Copilot",
    layout="wide",
)

st.title("Customer Support Copilot")
prompt_file = Path(__file__).resolve().parent / "prompts" / "system_prompt.txt"
system_prompt = prompt_file.read_text(encoding="utf-8") if prompt_file.is_file() else ""

question = st.text_area("Enter your question", height=150)

if st.button("Submit"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        full_prompt = f"{system_prompt.strip()}\n\nCustomer Question:\n\n{question.strip()}".strip()

        try:
            with st.spinner("Generating response..."):
                response = generate_response(full_prompt, model=MODEL_NAME)

            log_interaction(question.strip(), response, MODEL_NAME)
            st.subheader("Response")
            st.write(response)
        except requests.RequestException as error:
            st.error(f"Unable to reach Ollama: {error}")