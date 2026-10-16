from pathlib import Path

import streamlit as st
import requests

from llm.ollama_client import generate_response
from ingestion.document_manager import get_document_inventory, save_document
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

st.divider()
st.header("Document Upload")
uploaded_files = st.file_uploader(
    "Upload PDF, TXT, or Markdown files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
    key="support_documents",
)

if st.button("Save documents", disabled=not uploaded_files, key="save_documents"):
    for uploaded_file in uploaded_files:
        try:
            saved_document = save_document(uploaded_file)
            st.success(f"Uploaded {saved_document['document_name']}.")
        except (ValueError, OSError) as error:
            st.error(f"Could not upload {uploaded_file.name}: {error}")

st.subheader("Document Inventory")
st.dataframe(get_document_inventory(), width="stretch", hide_index=True)