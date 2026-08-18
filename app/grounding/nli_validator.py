from app.generation.models import GeneratedAnswer
from app.grounding.claim_extractor import ClaimExtractor
from app.grounding.models import ClaimVerification, GroundingResult
from app.grounding.validator import GroundingValidator
from app.grounding.verifier import ClaimVerifier
from app.retrieval.models import RetrievalResult


class NLIGroundingValidator(GroundingValidator):
    """Grounding validator using an NLI-based claim verifier."""

    def __init__(
        self,
        verifier: ClaimVerifier,
        claim_extractor: ClaimExtractor | None = None,
    ) -> None:
        self.verifier = verifier
        self.claim_extractor = claim_extractor or ClaimExtractor()

    def validate(
        self,
        answer: GeneratedAnswer,
        evidence: list[RetrievalResult],
    ) -> GroundingResult:
        claims = self.claim_extractor.extract(answer.answer)

        if not claims:
            return GroundingResult(
                grounded=False,
                claims=[],
            )

        evidence_by_content = {
            result.chunk.content: result.chunk.chunk_id
            for result in evidence
        }

        evidence_texts = list(evidence_by_content.keys())
        verifications: list[ClaimVerification] = []

        for claim in claims:
            matches = self.verifier.verify(
                claim=claim,
                evidence=evidence_texts,
            )

            supporting_chunk_ids = [
                evidence_by_content[text]
                for text, _score in matches
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

        grounded = all(
            verification.grounded
            for verification in verifications
        )

        return GroundingResult(
            grounded=grounded,
            claims=verifications,
            supporting_chunk_ids=supporting_chunk_ids,
        )
