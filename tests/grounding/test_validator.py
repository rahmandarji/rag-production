from app.generation.models import AnswerSource, GeneratedAnswer
from app.grounding.nli_validator import NLIGroundingValidator
from app.grounding.verifier import ClaimVerifier
from app.ingestion.models import Chunk
from app.retrieval.models import RetrievalResult


class FakeVerifier(ClaimVerifier):
    def __init__(self, supported: bool) -> None:
        self.supported = supported

    def verify(
        self,
        claim: str,
        evidence: list[str],
    ) -> list[tuple[str, float]]:
        if not self.supported:
            return []

        return [(evidence[0], 0.99)]


def make_evidence() -> list[RetrievalResult]:
    chunk = Chunk(
        chunk_id="doc-001:chunk-0001",
        document_id="doc-001",
        content="Path parameters are declared directly in the route path.",
        metadata={},
    )

    return [
        RetrievalResult(
            chunk=chunk,
            score=0.9,
        )
    ]


def test_validator_marks_supported_answer_as_grounded() -> None:
    answer = GeneratedAnswer(
        answer="Path parameters are declared directly in the route path."
    )

    validator = NLIGroundingValidator(
        verifier=FakeVerifier(supported=True)
    )

    result = validator.validate(
        answer=answer,
        evidence=make_evidence(),
    )

    assert result.grounded is True
    assert result.claims[0].grounded is True
    assert result.claims[0].supporting_chunk_ids == [
        "doc-001:chunk-0001"
    ]


def test_validator_rejects_unsupported_answer() -> None:
    answer = GeneratedAnswer(
        answer="FastAPI was created in 2018."
    )

    validator = NLIGroundingValidator(
        verifier=FakeVerifier(supported=False)
    )

    result = validator.validate(
        answer=answer,
        evidence=make_evidence(),
    )

    assert result.grounded is False
    assert result.claims[0].grounded is False
    assert result.claims[0].supporting_chunk_ids == []


def test_validator_rejects_empty_answer() -> None:
    answer = GeneratedAnswer(answer="")

    validator = NLIGroundingValidator(
        verifier=FakeVerifier(supported=True)
    )

    result = validator.validate(
        answer=answer,
        evidence=make_evidence(),
    )

    assert result.grounded is False
    assert result.claims == []
