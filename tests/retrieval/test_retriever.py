from unittest.mock import Mock

import pytest

from app.ingestion.models import Chunk
from app.retrieval.models import RetrievalResult
from app.retrieval.retriever import Retriever


def make_result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc-001",
            content=f"Content {chunk_id}",
        ),
        score=score,
    )


def test_retriever_without_reranker() -> None:
    embedding_provider = Mock()
    vector_store = Mock()

    embedding_provider.embed_query.return_value = [0.1, 0.2]

    vector_store.search.return_value = [
        make_result("chunk-1", 0.9),
        make_result("chunk-2", 0.8),
        make_result("chunk-3", 0.7),
    ]

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    results = retriever.search("test query", limit=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk-1"
    assert results[1].chunk.chunk_id == "chunk-2"

    vector_store.search.assert_called_once_with(
        query_embedding=[0.1, 0.2],
        limit=20,
    )


def test_retriever_uses_reranker() -> None:
    embedding_provider = Mock()
    vector_store = Mock()
    reranker = Mock()

    embedding_provider.embed_query.return_value = [0.1, 0.2]

    candidates = [
        make_result("chunk-1", 0.9),
        make_result("chunk-2", 0.8),
        make_result("chunk-3", 0.7),
    ]

    reranked = [
        candidates[2],
        candidates[0],
    ]

    vector_store.search.return_value = candidates
    reranker.rerank.return_value = reranked

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        reranker=reranker,
        retrieval_k=20,
    )

    results = retriever.search("test query", limit=2)

    assert results == reranked

    vector_store.search.assert_called_once_with(
        query_embedding=[0.1, 0.2],
        limit=20,
    )

    reranker.rerank.assert_called_once_with(
        query="test query",
        results=candidates,
        top_k=2,
    )


def test_retriever_rejects_invalid_limit() -> None:
    retriever = Retriever(
        embedding_provider=Mock(),
        vector_store=Mock(),
    )

    with pytest.raises(ValueError):
        retriever.search("test query", limit=0)


def test_retriever_rejects_invalid_retrieval_k() -> None:
    with pytest.raises(ValueError):
        Retriever(
            embedding_provider=Mock(),
            vector_store=Mock(),
            retrieval_k=0,
        )
