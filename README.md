# Customer Support Copilot

## Overview

A staged customer support assistant project. Stage 1 establishes the Streamlit interface and a minimal Ollama client.

## Architecture

Target flow: User -> Streamlit UI -> Ollama -> Phi3 Mini -> Response.

## Setup

Install dependencies with `pip install -r requirements.txt`, then start the app with `streamlit run app.py`.

The Ollama client expects Ollama to be available at `http://localhost:11434` with the `phi3:mini` model installed.

## Roadmap

- Stage 1: Project structure, Streamlit UI, and Ollama client foundation.
- Later stages: Extend the application with approved support capabilities.