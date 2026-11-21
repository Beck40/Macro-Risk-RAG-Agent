from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag_agent import RAGService
from src.rag_ingest import create_vector_db, ingest_pdf


app = FastAPI(title="Macro Risk PDF RAG Agent")


class QueryRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    file_path: str
    reset: bool = False


@app.get("/health")
def health():
    service = RAGService()
    return {
        "status": "ok",
        "vector_db_ready": service.vector_store_ready(),
    }


@app.post("/ingest")
def ingest(request: IngestRequest):
    if not request.file_path.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_path = Path(request.file_path)
    if not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="PDF file not found.")

    documents = ingest_pdf(str(pdf_path))
    if documents is None:
        raise HTTPException(status_code=400, detail="The PDF could not be processed.")

    chunks_indexed = create_vector_db(documents, reset=request.reset)
    if chunks_indexed == 0:
        raise HTTPException(status_code=500, detail="No chunks could be indexed.")

    return {
        "filename": pdf_path.name,
        "chunks_indexed": chunks_indexed,
    }


@app.post("/query")
def query(request: QueryRequest):
    service = RAGService()

    if not service.vector_store_ready():
        raise HTTPException(
            status_code=503,
            detail="Vector store is not available. Ingest a PDF first.",
        )
    if service.settings.groq_api_key is None:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured.",
        )

    try:
        return service.query_documents(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
