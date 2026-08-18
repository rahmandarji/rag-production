from app.generation.models import AnswerSource, GeneratedAnswer


def test_generated_answer_defaults_to_empty_sources() -> None:
    result = GeneratedAnswer(answer="Test answer")

    assert result.answer == "Test answer"
    assert result.sources == []


def test_answer_source_preserves_provenance() -> None:
    source = AnswerSource(
        chunk_id="doc-001:chunk-0001",
        document_id="doc-001",
        content="Path parameters are declared in the route.",
        metadata={"source": "guide.md"},
    )

    assert source.chunk_id == "doc-001:chunk-0001"
    assert source.document_id == "doc-001"
    assert source.metadata["source"] == "guide.md"


def test_generated_answer_contains_sources() -> None:
    source = AnswerSource(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="Evidence",
    )

    result = GeneratedAnswer(
        answer="Answer",
        sources=[source],
    )

    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == "chunk-1"
