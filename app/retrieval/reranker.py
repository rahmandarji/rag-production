from abc import ABC, abstractmethod

from app.retrieval.models import RetrievalResult


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Rerank retrieved candidates by query-document relevance."""
        raise NotImplementedError
