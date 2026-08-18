from unittest.mock import Mock

import pytest

from app.ingestion.models import Chunk
from app.retrieval.cross_encoder import CrossEncoderReranker
from app.retrieval.models import RetrievalResult


def make_result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc-001",
            content=f"Content for {chunk_id}",
        ),
        score=score,
    )


def test_cross_encoder_reranks_results() -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)

    reranker.model_name = "fake-model"
    reranker.model = Mock()

    reranker.model.predict.return_value = [0.2, 0.95, 0.4]

    results = [
        make_result("chunk-1", 0.9),
        make_result("chunk-2", 0.8),
        make_result("chunk-3", 0.7),
    ]

    ranked = reranker.rerank(
        query="test query",
        results=results,
        top_k=2,
    )

    assert len(ranked) == 2
    assert ranked[0].chunk.chunk_id == "chunk-2"
    assert ranked[1].chunk.chunk_id == "chunk-3"

    assert ranked[0].reranker_score == pytest.approx(0.95)
    assert ranked[1].reranker_score == pytest.approx(0.4)


def test_cross_encoder_handles_empty_results() -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)

    reranker.model_name = "fake-model"
    reranker.model = Mock()

    results = reranker.rerank(
        query="test query",
        results=[],
    )

    assert results == []


def test_cross_encoder_rejects_invalid_top_k() -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)

    reranker.model_name = "fake-model"
    reranker.model = Mock()

    with pytest.raises(ValueError):
        reranker.rerank(
            query="test",
            results=[],
            top_k=0,
        )
