import os
import shutil
from pathlib import Path
from typing import cast

import fitz
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings


def ingest_pdf(file_path):
    """Load and process a local PDF file."""
    try:
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found!")
            return None

        print(f"Loading PDF: {file_path}")

        documents = []
        with fitz.open(file_path) as pdf:
            for page_num in range(len(pdf)):
                pdf_page = pdf[page_num]
                text = cast(str, pdf_page.get_text())
                documents.append(Document(page_content=text, metadata={"source": file_path, "page": page_num + 1}))

        if not documents:
            print("Error: PDF appears to be empty!")
            return None

        print(f"Loaded {len(documents)} pages from PDF")
        return documents
    except Exception as exc:
        print(f"Error loading PDF: {str(exc)}")
        return None


def create_vector_db(documents, db_dir=None, reset=False) -> int:
    """Split documents and append them to a Chroma vector database."""
    try:
        print("\n--- PROCESSING DOCUMENTS ---")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=400)
        chunks = text_splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata.setdefault("page", 1)

        print(f"Created {len(chunks)} chunks")

        print("Initialising embedding model...")
        settings = get_settings()
        embedding_model = HuggingFaceEmbeddings(model_name=settings.embedding_model)

        directory = Path(db_dir) if db_dir is not None else settings.chroma_db_dir
        directory.mkdir(parents=True, exist_ok=True)

        if reset and directory.exists():
            shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

        vector_db = Chroma(persist_directory=str(directory), embedding_function=embedding_model)
        vector_db.add_documents(chunks)

        print(f"SUCCESS! Indexed {len(chunks)} chunks in '{directory}'")
        return len(chunks)
    except Exception as exc:
        print(f"Error creating database: {str(exc)}")
        return 0


def main():
    print("=" * 50)
    print("    FINANCIAL INSIGHT ENGINE - PDF INGESTION")
    print("=" * 50)
    
    file_path = input("\nEnter the PDF file path: ").strip()
    # Remove quotes if user included them
    file_path = file_path.strip('"').strip("'")
    
    documents = ingest_pdf(file_path)
    
    if documents:
        create_vector_db(documents)
    else:
        print("\nIngestion failed. Please try again.")

if __name__ == "__main__":
    main()
