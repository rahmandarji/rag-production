from abc import ABC, abstractmethod


class ClaimVerifier(ABC):
    @abstractmethod
    def verify(
        self,
        claim: str,
        evidence: list[str],
    ) -> list[tuple[str, float]]:
        """Return evidence/support scores for a claim."""
        raise NotImplementedError
