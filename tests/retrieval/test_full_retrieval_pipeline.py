from pathlib import Path

from app.ingestion.models import Chunk
from app.retrieval.cross_encoder import CrossEncoderReranker
from app.retrieval.embeddings import BGEEmbeddingProvider
from app.retrieval.qdrant_store import QdrantVectorStore
from app.retrieval.retriever import Retriever


def test_full_retrieval_and_reranking_pipeline(tmp_path: Path) -> None:
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
            content="OAuth allows applications to authenticate through an authorization provider.",
            metadata={"source": "guide.md"},
        ),
        Chunk(
            chunk_id="doc-001:chunk-0003",
            document_id="doc-001",
            content="FastAPI validates path parameters according to their declared types.",
            metadata={"source": "guide.md"},
        ),
    ]

    embedding_provider = BGEEmbeddingProvider()

    embeddings = embedding_provider.embed_documents(chunks)

    vector_store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="full_pipeline",
        vector_size=384,
    )

    vector_store.add(chunks, embeddings)

    reranker = CrossEncoderReranker()

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        reranker=reranker,
        retrieval_k=4,
    )

    results = retriever.search(
        "How does FastAPI handle path parameters?",
        limit=2,
    )

    assert len(results) == 2

    assert results[0].reranker_score is not None
    assert results[1].reranker_score is not None

    returned_ids = [result.chunk.chunk_id for result in results]

    assert "doc-001:chunk-0000" in returned_ids
    assert "doc-001:chunk-0003" in returned_ids

    assert results[0].reranker_score >= results[1].reranker_score
