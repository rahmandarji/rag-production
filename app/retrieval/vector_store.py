from abc import ABC, abstractmethod

from app.ingestion.models import Chunk
from app.retrieval.models import RetrievalResult


class VectorStore(ABC):
    @abstractmethod
    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunks and their embeddings."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[RetrievalResult]:
        """Return relevant chunks with similarity scores."""
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a document."""
        raise NotImplementedError
