import pytest

from app.ingestion.models import Chunk
from app.retrieval.embeddings import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    def embed_documents(self, chunks: list[Chunk]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in chunks]

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]


def test_embedding_provider_contract() -> None:
    provider = FakeEmbeddingProvider()

    chunks = [
        Chunk(
            chunk_id="doc-001:chunk-0000",
            document_id="doc-001",
            content="Test content.",
        )
    ]

    assert provider.embed_documents(chunks) == [[1.0, 0.0]]
    assert provider.embed_query("test") == [1.0, 0.0]


def test_embedding_provider_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        EmbeddingProvider()
