# Document Q&A Chatbot

A multi-document PDF question-answering app built with FastAPI, Streamlit, LangChain, FAISS, and Azure OpenAI.

## Overview

This project lets you:

- Upload one or more PDF documents
- Index them into a local FAISS vector store
- Select which uploaded PDFs should be searched
- Ask natural-language questions over the selected documents
- Get grounded answers with source snippets and page references

The app uses a LangChain tool-calling agent on top of Azure OpenAI so the model can decide when to search the selected documents before answering.

## Features

- Multi-document PDF upload and indexing
- Document scope selection from the Streamlit sidebar
- Azure OpenAI chat model integration
- Azure OpenAI embeddings with `text-embedding-ada-002`
- Local FAISS vector storage
- FastAPI backend with simple REST endpoints
- Streamlit frontend for fast testing and demos

## Tech Stack

- Backend: `FastAPI`
- Frontend: `Streamlit`
- LLM: `Azure OpenAI`
- Embeddings: `Azure OpenAI text-embedding-ada-002`
- Framework: `LangChain`
- Vector Store: `FAISS`
- PDF Parsing: `PyPDFLoader`
- Chunking: `RecursiveCharacterTextSplitter`

## Project Structure

```text
app/
  api/
  core/
  services/
data/
  documents/
  vectorstore/
streamlit_app.py
requirements.txt
```

## How It Works

1. A PDF is uploaded from the Streamlit UI.
2. The backend saves it into `data/documents`.
3. All uploaded PDFs are chunked and embedded.
4. Embeddings are stored in a local FAISS index.
5. In the UI, you choose which indexed PDFs to search.
6. A LangChain agent uses a retriever tool scoped only to those selected documents.
7. The final answer is returned with source references.

## Setup

1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies.
4. Add your Azure OpenAI configuration.
5. Start the backend and frontend.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## Environment Variables

Create a `.env` file in the project root.

```env
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-openai-key
OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_CHAT_DEPLOYMENT=your-chat-deployment-name
AZURE_OPENAI_CHAT_MODEL=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
RETRIEVAL_K=4
TEMPERATURE=0.1
MAX_RESPONSE_TOKENS=700
API_BASE_URL=http://localhost:8000/api
```

## Run the App

Start the backend:

```powershell
.\.venv\Scripts\python -m uvicorn app.main:app
```

Start the Streamlit app in a second terminal:

```powershell
.\.venv\Scripts\python -m streamlit run streamlit_app.py
```

Then open:

- FastAPI docs: `http://127.0.0.1:8000/docs`
- Streamlit UI: `http://localhost:8501`

## API Endpoints

- `GET /api/health`
- `GET /api/documents`
- `POST /api/documents/upload`
- `POST /api/documents/reindex`
- `POST /api/chat/ask`

## Example Flow

1. Start backend and frontend.
2. Upload one or more PDFs.
3. Wait for indexing to complete.
4. Select the PDFs you want to search in the sidebar.
5. Ask a question in the chat box.
6. Review the answer and cited sources.

## Notes

- This project currently supports PDF documents only.
- The FAISS index is stored locally in `data/vectorstore`.
- Uploaded PDFs are stored in `data/documents`.
- The app uses FAISS local deserialization for indexes it created itself.
- Uvicorn `--reload` may fail on some Windows setups; run without `--reload` if needed.


