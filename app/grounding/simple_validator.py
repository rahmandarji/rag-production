import re

from app.generation.models import GeneratedAnswer
from app.grounding.models import ClaimVerification, GroundingResult
from app.grounding.validator import GroundingValidator
from app.retrieval.models import RetrievalResult


class SimpleGroundingValidator(GroundingValidator):
    """Conservative baseline grounding validator.

    A claim is considered supported when its normalized text occurs
    in at least one evidence chunk.
    """

    def validate(
        self,
        generated_answer: GeneratedAnswer,
        evidence: list[RetrievalResult],
    ) -> GroundingResult:
        answer = generated_answer.answer.strip()

        if not answer or not evidence:
            return GroundingResult(
                grounded=False,
                answer=answer,
            )

        claims = self._extract_claims(answer)

        verifications: list[ClaimVerification] = []

        for claim in claims:
            normalized_claim = self._normalize(claim)
            supporting_chunk_ids = [
                result.chunk.chunk_id
                for result in evidence
                if normalized_claim in self._normalize(result.chunk.content)
            ]

            verifications.append(
                ClaimVerification(
                    claim=claim,
                    grounded=bool(supporting_chunk_ids),
                    supporting_chunk_ids=supporting_chunk_ids,
                )
            )

        supporting_chunk_ids = sorted(
            {
                chunk_id
                for verification in verifications
                for chunk_id in verification.supporting_chunk_ids
            }
        )

        grounded = bool(verifications) and all(
            verification.grounded for verification in verifications
        )

        return GroundingResult(
            grounded=grounded,
            answer=answer,
            claims=verifications,
            supporting_chunk_ids=supporting_chunk_ids,
        )

    @staticmethod
    def _extract_claims(answer: str) -> list[str]:
        return [
            claim.strip()
            for claim in re.split(r"[.!?]+", answer)
            if claim.strip()
        ]

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())
