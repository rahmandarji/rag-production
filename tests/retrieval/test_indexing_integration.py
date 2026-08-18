from pathlib import Path

from app.ingestion.service import IngestionService
from app.retrieval.embeddings import BGEEmbeddingProvider
from app.retrieval.indexer import RetrievalIndexer
from app.retrieval.qdrant_store import QdrantVectorStore
from app.retrieval.retriever import Retriever


def test_ingestion_to_real_retrieval_pipeline(tmp_path: Path) -> None:
    document_path = tmp_path / "guide.md"

    document_path.write_text(
        "# Authentication\n\n"
        "API keys are passed using the appropriate request header.\n\n"
        "# Routing\n\n"
        "Path parameters are declared directly in the route path.\n\n"
        "# OAuth\n\n"
        "OAuth allows applications to authenticate through an authorization provider.",
        encoding="utf-8",
    )

    ingestion = IngestionService()

    document = ingestion.ingest_file(document_path)
    chunks = ingestion.chunk_document(document)

    assert chunks
    assert all(chunk.document_id == document.document_id for chunk in chunks)

    embedding_provider = BGEEmbeddingProvider()

    vector_store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="ingestion_retrieval_test",
        vector_size=384,
    )

    indexer = RetrievalIndexer(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    indexer.index(chunks)

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        retrieval_k=3,
    )

    results = retriever.search(
        "How are path parameters declared?",
        limit=2,
    )

    assert results
    assert results[0].chunk.document_id == document.document_id
    assert "Path parameters" in results[0].chunk.content
