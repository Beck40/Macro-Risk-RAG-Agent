from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from pydantic import SecretStr

from src.api import app


client = TestClient(app)


def test_health_returns_status_without_live_configuration():
    with patch("src.api.RAGService") as service_class:
        service = MagicMock()
        service.vector_store_ready.return_value = False
        service_class.return_value = service

        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "vector_db_ready": False}


def test_query_uses_mocked_rag_service():
    with patch("src.api.RAGService") as service_class:
        service = MagicMock()
        service.vector_store_ready.return_value = True
        service.settings.groq_api_key = SecretStr("test-key")
        service.query_documents.return_value = {
            "answer": "Mocked answer",
            "sources": [{"source": "report.pdf", "page": 3}],
        }
        service_class.return_value = service

        response = client.post("/query", json={"question": "What is the risk?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Mocked answer"
    assert response.json()["sources"] == [{"source": "report.pdf", "page": 3}]
    service.query_documents.assert_called_once_with("What is the risk?")


def test_ingest_indexes_valid_pdf(tmp_path):
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    with patch("src.api.ingest_pdf", return_value=["document"]), patch("src.api.create_vector_db", return_value=12) as create_db:
        response = client.post("/ingest", json={"file_path": str(pdf_path), "reset": False})

    assert response.status_code == 200
    assert response.json() == {"filename": "report.pdf", "chunks_indexed": 12}
    create_db.assert_called_once_with(["document"], reset=False)


def test_ingest_rejects_non_pdf_paths():
    response = client.post("/ingest", json={"file_path": "notes.txt"})

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_ingest_returns_404_for_missing_pdf(tmp_path):
    missing_pdf = tmp_path / "missing.pdf"

    response = client.post("/ingest", json={"file_path": str(missing_pdf)})

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
