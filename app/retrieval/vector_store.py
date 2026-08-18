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
