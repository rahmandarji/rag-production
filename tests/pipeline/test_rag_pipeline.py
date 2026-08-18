from app.generation.models import GeneratedAnswer
from app.generation.provider import GenerationProvider
from app.grounding.models import ClaimVerification, GroundingResult
from app.grounding.validator import GroundingValidator
from app.ingestion.models import Chunk
from app.pipeline.rag_pipeline import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    RAGPipeline,
)
from app.retrieval.models import RetrievalResult


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = 0

    def search(self, query: str, limit: int = 5):
        self.calls += 1
        return self.results[:limit]


class FakeGenerator(GenerationProvider):
    def __init__(self):
        self.calls = 0

    def generate(self, query, evidence):
        self.calls += 1

        return GeneratedAnswer(
            answer="Path parameters are declared directly in the route path."
        )


class FakeValidator(GroundingValidator):
    def __init__(self, grounded: bool):
        self.grounded = grounded
        self.calls = 0

    def validate(self, answer, evidence):
        self.calls += 1

        supporting_ids = (
            [evidence[0].chunk.chunk_id]
            if self.grounded and evidence
            else []
        )

        claims = (
            [
                ClaimVerification(
                    claim=answer.answer,
                    grounded=True,
                    supporting_chunk_ids=supporting_ids,
                )
            ]
            if self.grounded
            else [
                ClaimVerification(
                    claim=answer.answer,
                    grounded=False,
                    supporting_chunk_ids=[],
                )
            ]
        )

        return GroundingResult(
            grounded=self.grounded,
            claims=claims,
            supporting_chunk_ids=supporting_ids,
        )


def make_evidence():
    return [
        RetrievalResult(
            chunk=Chunk(
                chunk_id="doc-001:chunk-0001",
                document_id="doc-001",
                content="Path parameters are declared directly in the route path.",
                metadata={"source": "guide.md"},
            ),
            score=0.9,
        ),
        RetrievalResult(
            chunk=Chunk(
                chunk_id="doc-001:chunk-0002",
                document_id="doc-001",
                content="FastAPI validates path parameters according to their declared types.",
                metadata={"source": "guide.md"},
            ),
            score=0.8,
        ),
    ]


def test_pipeline_generates_and_validates_answer():
    retriever = FakeRetriever(make_evidence())
    generator = FakeGenerator()
    validator = FakeValidator(grounded=True)

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        grounding_validator=validator,
    )

    result = pipeline.query("How do path parameters work?")

    assert result.answer.answer == (
        "Path parameters are declared directly in the route path."
    )
    assert result.grounding.grounded is True
    assert len(result.evidence) == 2

    assert len(result.answer.sources) == 1
    assert result.answer.sources[0].chunk_id == "doc-001:chunk-0001"


def test_pipeline_rejects_ungrounded_generation():
    retriever = FakeRetriever(make_evidence())
    generator = FakeGenerator()
    validator = FakeValidator(grounded=False)

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        grounding_validator=validator,
    )

    result = pipeline.query("Who founded the company?")

    assert result.answer.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert result.answer.sources == []
    assert result.grounding.grounded is False
    assert result.evidence == make_evidence()


def test_pipeline_handles_empty_retrieval_without_generation():
    retriever = FakeRetriever([])
    generator = FakeGenerator()
    validator = FakeValidator(grounded=True)

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        grounding_validator=validator,
    )

    result = pipeline.query("Who founded Company X?")

    assert result.answer.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert result.answer.sources == []
    assert result.evidence == []
    assert result.grounding.grounded is False

    assert retriever.calls == 1
    assert generator.calls == 0
    assert validator.calls == 0


def test_pipeline_rejects_empty_query():
    retriever = FakeRetriever([])
    generator = FakeGenerator()
    validator = FakeValidator(grounded=True)

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        grounding_validator=validator,
    )

    try:
        pipeline.query("   ")
        assert False
    except ValueError as exc:
        assert str(exc) == "Query must not be empty."

    assert retriever.calls == 0


def test_pipeline_rejects_invalid_retrieval_limit():
    pipeline = RAGPipeline(
        retriever=FakeRetriever(make_evidence()),
        generator=FakeGenerator(),
        grounding_validator=FakeValidator(grounded=True),
    )

    try:
        pipeline.query("test", retrieval_limit=0)
        assert False
    except ValueError as exc:
        assert str(exc) == "retrieval_limit must be greater than 0."


def test_pipeline_only_returns_supporting_sources():
    retriever = FakeRetriever(make_evidence())
    generator = FakeGenerator()
    validator = FakeValidator(grounded=True)

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        grounding_validator=validator,
    )

    result = pipeline.query("How do path parameters work?")

    source_ids = [source.chunk_id for source in result.answer.sources]

    assert source_ids == ["doc-001:chunk-0001"]
