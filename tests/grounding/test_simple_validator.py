from app.generation.models import GeneratedAnswer
from app.grounding.simple_validator import SimpleGroundingValidator
from app.retrieval.models import RetrievalResult
from app.ingestion.models import Chunk


def make_result(chunk_id: str, content: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc-001",
            content=content,
            metadata={"source": "guide.md"},
        ),
        score=0.9,
    )


def test_supported_answer_is_grounded() -> None:
    evidence = [
        make_result(
            "doc-001:chunk-0001",
            "Path parameters are declared directly in the route path.",
        )
    ]

    answer = GeneratedAnswer(
        answer="Path parameters are declared directly in the route path."
    )

    result = SimpleGroundingValidator().validate(answer, evidence)

    assert result.grounded is True
    assert result.claims[0].grounded is True
    assert result.supporting_chunk_ids == ["doc-001:chunk-0001"]


def test_unsupported_answer_is_not_grounded() -> None:
    evidence = [
        make_result(
            "doc-001:chunk-0001",
            "Path parameters are declared directly in the route path.",
        )
    ]

    answer = GeneratedAnswer(
        answer="FastAPI was created in 2018."
    )

    result = SimpleGroundingValidator().validate(answer, evidence)

    assert result.grounded is False
    assert result.claims[0].grounded is False
    assert result.supporting_chunk_ids == []


def test_mixed_answer_is_not_grounded() -> None:
    evidence = [
        make_result(
            "doc-001:chunk-0001",
            "Path parameters are declared directly in the route path.",
        )
    ]

    answer = GeneratedAnswer(
        answer=(
            "Path parameters are declared directly in the route path. "
            "FastAPI was created in 2018."
        )
    )

    result = SimpleGroundingValidator().validate(answer, evidence)

    assert result.grounded is False
    assert len(result.claims) == 2
    assert result.claims[0].grounded is True
    assert result.claims[1].grounded is False


def test_empty_evidence_is_not_grounded() -> None:
    answer = GeneratedAnswer(answer="Some answer.")

    result = SimpleGroundingValidator().validate(answer, [])

    assert result.grounded is False
    assert result.claims == []


def test_paraphrased_answer_is_not_supported_by_exact_matching() -> None:
    evidence = [
        make_result(
            "doc-001:chunk-0001",
            "Path parameters are declared directly in the route path.",
        )
    ]

    answer = GeneratedAnswer(
        answer="You define path parameters inside the route."
    )

    result = SimpleGroundingValidator().validate(answer, evidence)

    assert result.grounded is False
