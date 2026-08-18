from pathlib import Path

from app.ingestion.models import Chunk
from app.retrieval.embeddings import BGEEmbeddingProvider
from app.retrieval.qdrant_store import QdrantVectorStore
from app.retrieval.retriever import Retriever


def test_real_embedding_to_qdrant_retrieval(tmp_path: Path) -> None:
    chunks = [
        Chunk(
            chunk_id="doc-001:chunk-0000",
            document_id="doc-001",
            content="Path parameters are declared directly in the route path.",
            metadata={"source": "guide.md"},
        ),
        Chunk(
            chunk_id="doc-001:chunk-0001",
            document_id="doc-001",
            content="API keys are passed using the appropriate request header.",
            metadata={"source": "guide.md"},
        ),
        Chunk(
            chunk_id="doc-001:chunk-0002",
            document_id="doc-001",
            content="OAuth allows applications to authenticate through an "
            "authorization provider.",
            metadata={"source": "guide.md"},
        ),
    ]

    embedding_provider = BGEEmbeddingProvider()

    embeddings = embedding_provider.embed_documents(chunks)

    vector_store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="integration_test",
        vector_size=384,
    )

    vector_store.add(chunks, embeddings)

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    results = retriever.search(
        "How do I define parameters inside a URL route?",
        limit=2,
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "doc-001:chunk-0000"
    assert "Path parameters" in results[0].chunk.content
