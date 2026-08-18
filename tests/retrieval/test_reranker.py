import pytest

from app.ingestion.models import Chunk
from app.retrieval.models import RetrievalResult
from app.retrieval.reranker import Reranker


class FakeReranker(Reranker):
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        return sorted(
            results,
            key=lambda result: result.score,
            reverse=True,
        )[:top_k]


def make_result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc-001",
            content=f"Content for {chunk_id}",
        ),
        score=score,
    )


def test_reranker_returns_top_k() -> None:
    reranker = FakeReranker()

    results = [
        make_result("chunk-1", 0.2),
        make_result("chunk-2", 0.9),
        make_result("chunk-3", 0.5),
    ]

    ranked = reranker.rerank(
        query="test query",
        results=results,
        top_k=2,
    )

    assert len(ranked) == 2
    assert ranked[0].chunk.chunk_id == "chunk-2"
    assert ranked[1].chunk.chunk_id == "chunk-3"


def test_reranker_rejects_invalid_top_k() -> None:
    reranker = FakeReranker()

    with pytest.raises(ValueError):
        reranker.rerank(
            query="test",
            results=[],
            top_k=0,
        )
