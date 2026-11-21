import os
from functools import lru_cache
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import get_settings


@lru_cache
def get_embedding_model(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)


class RAGService:
    def __init__(self, db_dir=None):
        self.settings = get_settings()
        self.db_dir = Path(db_dir) if db_dir is not None else self.settings.chroma_db_dir

    def vector_store_ready(self) -> bool:
        if not self.db_dir.is_dir():
            return False
        return (self.db_dir / "chroma.sqlite3").is_file()

    def query_documents(self, question: str) -> dict:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")
        if self.settings.groq_api_key is None:
            raise ValueError("GROQ_API_KEY is not set.")
        if not self.vector_store_ready():
            raise FileNotFoundError("Vector store is not available.")

        embedding_model = get_embedding_model(self.settings.embedding_model)
        vector_db = Chroma(
            persist_directory=str(self.db_dir),
            embedding_function=embedding_model,
        )
        llm = ChatGroq(
            model=self.settings.llm_model,
            temperature=0,
            api_key=self.settings.groq_api_key,
        )

        prompt = ChatPromptTemplate.from_template(
            """You are a Senior Financial Analyst with expertise in analyzing financial documents.
Analyze the provided context carefully and answer the question in detail.
If the answer is not in the context, say "I cannot find that information in the documents."

CONTEXT:
{context}

QUESTION:
{question}

DETAILED ANSWER:"""
        )
        retriever = vector_db.as_retriever(search_kwargs={"k": 7})

        def format_documents(documents):
            return "\n\n".join(document.page_content for document in documents)

        documents = retriever.invoke(question)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke(
            {
                "context": format_documents(documents),
                "question": question,
            }
        )

        sources = []
        seen_sources = set()
        for document in documents:
            source = str(document.metadata.get("source", "Unknown"))
            page = document.metadata.get("page")
            filename = os.path.basename(source) if "://" not in source else source
            citation = {"source": filename}
            if page is not None:
                citation["page"] = int(page)
            key = (citation["source"], citation.get("page"))
            if key not in seen_sources:
                seen_sources.add(key)
                sources.append(citation)

        return {"answer": answer, "sources": sources}


def query_documents(question: str, db_dir=None) -> dict:
    return RAGService(db_dir=db_dir).query_documents(question)


def start_rag_agent():
    print("\n" + "=" * 50)
    print(" FINANCIAL INSIGHT ENGINE (Ready)")
    print("=" * 50)
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Query: ")
        if query.lower() in ["exit", "quit"]:
            break

        print("\n Searching documents...")
        try:
            result = query_documents(query)
            print(f"\nAgent: {result['answer']}\n")
            print("--- Sources ---")
            for source in result["sources"]:
                citation = f" {source['source']}"
                if source.get("page") is not None:
                    citation += f" (Page {source['page']})"
                print(citation)
            print("-" * 20 + "\n")
        except Exception as exc:
            print(f" Error: {exc}")


if __name__ == "__main__":
    start_rag_agent()
