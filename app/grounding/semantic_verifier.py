from abc import ABC, abstractmethod


class SemanticVerifier(ABC):
    @abstractmethod
    def verify(
        self,
        claim: str,
        evidence: str,
    ) -> float:
        """Return a semantic support score between 0 and 1."""
        raise NotImplementedError
