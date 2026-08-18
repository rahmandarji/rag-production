from pathlib import Path

from app.generation.models import GeneratedAnswer
from app.grounding.nli_validator import NLIGroundingValidator
from app.grounding.nli_verifier import NLIVerifier
from app.retrieval.cross_encoder import CrossEncoderReranker
from app.retrieval.embeddings import BGEEmbeddingProvider
from app.retrieval.models import RetrievalResult
from app.retrieval.qdrant_store import QdrantVectorStore
from app.retrieval.retriever import Retriever


def test_real_retrieval_and_grounding_smoke(tmp_path: Path) -> None:
    chunks = [
        {
            "chunk_id": "doc-001:chunk-0000",
            "document_id": "doc-001",
            "content": "Path parameters are declared directly in the route path.",
            "metadata": {"source": "guide.md"},
        },
        {
            "chunk_id": "doc-001:chunk-0001",
            "document_id": "doc-001",
            "content": "API keys are passed using the appropriate request header.",
            "metadata": {"source": "guide.md"},
        },
        {
            "chunk_id": "doc-001:chunk-0002",
            "document_id": "doc-001",
            "content": "OAuth allows applications to authenticate through an authorization provider.",
            "metadata": {"source": "guide.md"},
        },
        {
            "chunk_id": "doc-001:chunk-0003",
            "document_id": "doc-001",
            "content": "FastAPI validates path parameters according to their declared types.",
            "metadata": {"source": "guide.md"},
        },
    ]

    from app.ingestion.models import Chunk

    chunk_models = [Chunk(**chunk) for chunk in chunks]

    embedding_provider = BGEEmbeddingProvider()

    embeddings = embedding_provider.embed_documents(chunk_models)

    vector_store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="smoke_test",
        vector_size=384,
    )

    vector_store.add(
        chunk_models,
        embeddings,
    )

    reranker = CrossEncoderReranker()

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        reranker=reranker,
        retrieval_k=4,
    )

    results = retriever.search(
        query="How does FastAPI handle path parameters?",
        limit=2,
    )

    assert len(results) == 2

    assert all(
        isinstance(result, RetrievalResult)
        for result in results
    )

    assert all(
        result.reranker_score is not None
        for result in results
    )

    returned_ids = {
        result.chunk.chunk_id
        for result in results
    }

    assert "doc-001:chunk-0000" in returned_ids
    assert "doc-001:chunk-0003" in returned_ids

    answer = GeneratedAnswer(
        answer="Path parameters are declared directly in the route path.",
        sources=[],
    )

    validator = NLIGroundingValidator(
        verifier=NLIVerifier(),
    )

    grounding = validator.validate(
        answer=answer,
        evidence=results,
    )

    assert grounding.grounded is True

    assert "doc-001:chunk-0000" in grounding.supporting_chunk_ids
