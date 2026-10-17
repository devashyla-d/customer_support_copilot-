from pathlib import Path

import streamlit as st
import requests
import pandas as pd

from config import MODEL_NAME
from embeddings.embedding_service import (
    generate_embeddings_for_document,
    get_embedding_inventory,
)
from llm.ollama_client import generate_response_stream
from ingestion.chunker import chunk_document, get_saved_chunks, read_document_text, save_chunks
from ingestion.document_manager import get_document_inventory, save_document
from logs.logger import log_interaction
from vectordb.chroma_service import (
    get_collection_inventory,
    index_embeddings,
    search_documents,
)

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

        st.subheader("Response")
        response_placeholder = st.empty()
        response_chunks = []

        try:
            with st.spinner("Generating response..."):
                for chunk in generate_response_stream(full_prompt, model=MODEL_NAME):
                    response_chunks.append(chunk)
                    response_placeholder.markdown("".join(response_chunks))

            response = "".join(response_chunks)
            log_interaction(question.strip(), response, MODEL_NAME)
            response_placeholder.markdown(response)
        except requests.RequestException:
            response_placeholder.empty()
            st.error("Unable to reach Ollama")

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
document_inventory = get_document_inventory()
st.dataframe(document_inventory, width="stretch", hide_index=True)

st.divider()
st.header("Generate Chunks")
document_names = document_inventory["document_name"].dropna().astype(str).unique().tolist()

if not document_names:
    st.info("Upload a document before generating chunks.")
else:
    selected_document = st.selectbox("Select uploaded document", document_names)

    if st.button("Generate Chunks", key="generate_chunks"):
        try:
            document_text = read_document_text(selected_document)
            generated_chunks = chunk_document(selected_document, document_text)
            save_chunks(generated_chunks)
            st.success(f"Generated and saved {len(generated_chunks)} chunks.")
        except (FileNotFoundError, OSError, ValueError) as error:
            st.error(f"Could not generate chunks: {error}")

    saved_chunks = get_saved_chunks(selected_document)
    if saved_chunks.empty:
        st.info("No chunks have been generated for this document yet.")
    else:
        display_chunks = saved_chunks[
            ["document_name", "chunk_number", "chunk_text"]
        ].rename(
            columns={
                "document_name": "Document Name",
                "chunk_number": "Chunk Number",
                "chunk_text": "Chunk Text",
            }
        )
        st.dataframe(display_chunks, width="stretch", hide_index=True)

st.divider()
st.header("Embeddings")
chunk_inventory = get_saved_chunks()
chunk_document_names = (
    chunk_inventory["document_name"].dropna().astype(str).unique().tolist()
)

if not chunk_document_names:
    st.info("Generate document chunks before creating embeddings.")
else:
    selected_embedding_document = st.selectbox(
        "Select document for embeddings",
        chunk_document_names,
        key="embedding_document",
    )

    if st.button("Generate embeddings", key="generate_embeddings"):
        try:
            with st.spinner("Generating embeddings..."):
                generated_embeddings = generate_embeddings_for_document(
                    selected_embedding_document
                )
            st.success(f"Generated {len(generated_embeddings)} embeddings.")
        except (FileNotFoundError, OSError, ValueError) as error:
            st.error(f"Could not generate embeddings: {error}")

    embedding_inventory = get_embedding_inventory(selected_embedding_document)
    if embedding_inventory.empty:
        st.info("No embeddings have been generated for this document yet.")
    else:
        st.dataframe(embedding_inventory, width="stretch", hide_index=True)

st.divider()
st.header("Semantic Search")
st.caption("Index generated embeddings before searching. Reindexing safely updates changed chunks.")

if st.button("Index / Reindex Embeddings", key="index_chroma_embeddings"):
    try:
        with st.spinner("Indexing embeddings in ChromaDB..."):
            indexed_count = index_embeddings()
        st.success(f"Indexed {indexed_count} chunks in ChromaDB.")
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(f"Could not index embeddings: {error}")

semantic_question = st.text_input(
    "Enter a question to search indexed documents",
    key="semantic_search_question",
)

if st.button("Search", key="semantic_search"):
    if not semantic_question.strip():
        st.warning("Enter a question to search.")
    else:
        try:
            with st.spinner("Searching indexed chunks..."):
                search_results = search_documents(semantic_question, top_k=5)

            if search_results:
                display_results = pd.DataFrame(search_results)[
                    ["document_name", "chunk_number", "chunk_text", "similarity_score"]
                ].rename(
                    columns={
                        "document_name": "Document Name",
                        "chunk_number": "Chunk Number",
                        "chunk_text": "Chunk Text",
                        "similarity_score": "Similarity Score",
                    }
                )
                st.dataframe(display_results, width="stretch", hide_index=True)
            else:
                st.info("No indexed chunks found. Index embeddings before searching.")
        except (FileNotFoundError, OSError, ValueError) as error:
            st.error(f"Could not search documents: {error}")

st.divider()
st.header("ChromaDB Collection")
st.caption("Read-only view of the persistent enterprise_support_docs collection.")
collection_limit = st.number_input(
    "Maximum records to display",
    min_value=1,
    max_value=1000,
    value=100,
    step=50,
    key="chroma_collection_limit",
)

if st.button("Refresh collection view", key="refresh_chroma_collection"):
    try:
        with st.spinner("Loading collection records..."):
            collection_count, collection_records = get_collection_inventory(
                int(collection_limit)
            )
        st.session_state["chroma_collection_snapshot"] = {
                "count": collection_count,
                "records": collection_records,
            }
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"Could not load the ChromaDB collection: {error}")

collection_snapshot = st.session_state.get("chroma_collection_snapshot")
if collection_snapshot is None:
    st.info("Refresh the view to inspect the current ChromaDB collection.")
else:
    st.metric("Indexed chunks", collection_snapshot["count"])
    collection_records = collection_snapshot["records"]
    if not collection_records:
        st.info("The collection is empty. Index embeddings before viewing records.")
    else:
        collection_table = pd.DataFrame(collection_records)[
            [
                "chunk_id",
                "document_name",
                "chunk_number",
                "chunk_text",
                "embedding_dimensions",
            ]
        ].rename(
            columns={
                "chunk_id": "Chunk ID",
                "document_name": "Document Name",
                "chunk_number": "Chunk Number",
                "chunk_text": "Chunk Text",
                "embedding_dimensions": "Vector Dimensions",
            }
        )
        st.dataframe(collection_table, width="stretch", hide_index=True)

        selected_chunk_id = st.selectbox(
            "Inspect stored vector",
            [record["chunk_id"] for record in collection_records],
            key="inspect_chroma_vector",
        )
        selected_record = next(
            record
            for record in collection_records
            if record["chunk_id"] == selected_chunk_id
        )
        with st.expander("Show raw embedding vector"):
            st.json(selected_record["embedding_vector"])