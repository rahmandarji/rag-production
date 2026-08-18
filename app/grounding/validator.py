from abc import ABC, abstractmethod

from app.generation.models import GeneratedAnswer
from app.grounding.models import GroundingResult
from app.retrieval.models import RetrievalResult


class GroundingValidator(ABC):
    """Interface for validating whether a generated answer is grounded."""

    @abstractmethod
    def validate(
        self,
        answer: GeneratedAnswer,
        evidence: list[RetrievalResult],
    ) -> GroundingResult:
        """Validate a generated answer against supplied evidence."""
        raise NotImplementedError
