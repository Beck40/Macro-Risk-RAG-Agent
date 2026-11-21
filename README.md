# Macro Risk RAG Agent

## Objective

Macroeconomic research is valuable only when analysts can retrieve evidence quickly and cite it. This repository provides a PDF retrieval-augmented agent for risk committees, portfolio managers and scenario teams. It ingests financial PDFs, retrieves relevant passages and produces source-aware answers while preserving page-level traceability.

## Architecture

```text
Macro-Risk-RAG-Agent/
├── .env.example
├── .github/
│   └── workflows/
│       └── ci.yml                     # pytest checks for pushes and pull requests
├── data/
│   └── .gitkeep
├── docs/
│   └── GenAI Financial Analyst - Technical Guide.pdf
├── infrastructure/
│   └── Dockerfile                     # Azure Container Apps-compatible image
├── src/
│   ├── api.py                         # FastAPI health, ingest and query endpoints
│   ├── config.py                      # Central pydantic-settings configuration
│   ├── rag_ingest.py                  # PDF extraction and vector indexing
│   ├── rag_agent.py                   # Retrieval chain and source citations
│   └── __init__.py
└── tests/
    └── test_api.py
```

## Environment setup

Create or update a local `.env` from `.env.example`. Docker and Azure environment variables take precedence over `.env` values.

```env
GROQ_API_KEY=your_groq_api_key
CHROMA_DB_DIR=data/chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=llama-3.1-8b-instant
```

## Quickstart

Install dependencies and start the API locally:

```bash
pip install -r requirements.txt
uvicorn src.api:app --reload
```

Ingest a PDF directly from a local path:

```bash
python -m src.rag_ingest
```

Build and run the container:

```bash
docker build -f infrastructure/Dockerfile -t macro-risk-rag .
docker run --rm -p 8000:8000 --env-file .env macro-risk-rag
```

The API is available at `http://localhost:8000`.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Reports API status and whether a vector store is available |
| `POST` | `/ingest` | Accepts `{"file_path": ".../report.pdf", "reset": false}` and indexes the PDF |
| `POST` | `/query` | Answers a question using retrieved PDF context and page citations |
