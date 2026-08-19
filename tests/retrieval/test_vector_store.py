from app.ingestion.models import Chunk
from app.retrieval.models import RetrievalResult
from app.retrieval.vector_store import VectorStore


class FakeVectorStore(VectorStore):
    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        pass

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[RetrievalResult]:
        return []

    def delete_document(self, document_id: str) -> None:
        pass


def test_vector_store_contract() -> None:
    store = FakeVectorStore()

    assert isinstance(store, VectorStore)
