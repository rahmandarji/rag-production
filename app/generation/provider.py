from abc import ABC, abstractmethod

from app.generation.models import GeneratedAnswer
from app.retrieval.models import RetrievalResult


class GenerationProvider(ABC):
    @abstractmethod
    def generate(
        self,
        query: str,
        evidence: list[RetrievalResult],
    ) -> GeneratedAnswer:
        """Generate an answer using only the supplied evidence."""
        raise NotImplementedError
