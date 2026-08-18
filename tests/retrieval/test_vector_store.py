import pytest

from app.ingestion.models import Chunk
from app.retrieval.vector_store import VectorStore


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[Chunk]:
        return self.chunks[:limit]


def test_vector_store_contract() -> None:
    store = FakeVectorStore()

    chunks = [
        Chunk(
            chunk_id="doc-001:chunk-0000",
            document_id="doc-001",
            content="Path parameters.",
        )
    ]

    store.add(chunks, [[1.0, 0.0]])

    results = store.search([1.0, 0.0], limit=1)

    assert len(results) == 1
    assert results[0].content == "Path parameters."


def test_vector_store_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        VectorStore()
