from fastapi.testclient import TestClient

from app.main import app
from app.pipeline.rag_pipeline import RAGResponse
from app.generation.models import AnswerSource, GeneratedAnswer
from app.grounding.models import GroundingResult
from app.retrieval.models import RetrievalResult


class FakePipeline:
    def query(self, query: str, retrieval_limit: int = 5) -> RAGResponse:
        return RAGResponse(
            answer=GeneratedAnswer(
                answer="Path parameters are declared in the route path.",
                sources=[
                    AnswerSource(
                        chunk_id="doc-001:chunk-0001",
                        document_id="doc-001",
                        content="Path parameters are declared in the route path.",
                        metadata={"source": "guide.md"},
                    )
                ],
            ),
            grounding=GroundingResult(
                grounded=True,
                claims=[],
                supporting_chunk_ids=["doc-001:chunk-0001"],
            ),
            evidence=[],
        )


def test_query_endpoint_returns_grounded_answer() -> None:
    app.state.rag_pipeline = FakePipeline()

    client = TestClient(app)

    response = client.post(
        "/api/v1/query",
        json={
            "query": "How do path parameters work?",
            "retrieval_limit": 5,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == (
        "Path parameters are declared in the route path."
    )
    assert body["grounded"] is True
    assert len(body["sources"]) == 1
    assert body["sources"][0]["chunk_id"] == "doc-001:chunk-0001"
    assert body["sources"][0]["document_id"] == "doc-001"


def test_query_endpoint_rejects_empty_query() -> None:
    app.state.rag_pipeline = FakePipeline()

    client = TestClient(app)

    response = client.post(
        "/api/v1/query",
        json={"query": "   "},
    )

    assert response.status_code == 422


def test_query_endpoint_rejects_invalid_retrieval_limit() -> None:
    app.state.rag_pipeline = FakePipeline()

    client = TestClient(app)

    response = client.post(
        "/api/v1/query",
        json={
            "query": "How do path parameters work?",
            "retrieval_limit": 0,
        },
    )

    assert response.status_code == 422


def test_app_lifespan_creates_rag_pipeline() -> None:
    from unittest.mock import patch

    fake_pipeline = FakePipeline()

    with patch("app.main.create_rag_pipeline", return_value=fake_pipeline):
        with TestClient(app):
            assert app.state.rag_pipeline is fake_pipeline


def test_query_endpoint_with_real_rag_pipeline() -> None:
    from unittest.mock import Mock

    from app.ingestion.models import Chunk
    from app.pipeline.rag_pipeline import RAGPipeline

    chunk = Chunk(
        chunk_id="doc-001:chunk-0001",
        document_id="doc-001",
        content="Path parameters are declared in the route path.",
        metadata={"source": "guide.md"},
    )

    retrieval_result = RetrievalResult(
        chunk=chunk,
        score=0.95,
    )

    retriever = Mock()
    retriever.search.return_value = [retrieval_result]

    generator = Mock()
    generator.generate.return_value = GeneratedAnswer(
        answer="Path parameters are declared in the route path.",
        sources=[
            AnswerSource(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                metadata=chunk.metadata,
            )
        ],
    )

    grounding_validator = Mock()
    grounding_validator.validate.return_value = GroundingResult(
        grounded=True,
        claims=[],
        supporting_chunk_ids=[chunk.chunk_id],
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        grounding_validator=grounding_validator,
    )

    app.state.rag_pipeline = pipeline

    client = TestClient(app)

    response = client.post(
        "/api/v1/query",
        json={
            "query": "How are path parameters declared?",
            "retrieval_limit": 5,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == "Path parameters are declared in the route path."
    assert body["grounded"] is True
    assert body["sources"][0]["chunk_id"] == chunk.chunk_id

    retriever.search.assert_called_once_with(
        query="How are path parameters declared?",
        limit=5,
    )

    generator.generate.assert_called_once()

    grounding_validator.validate.assert_called_once()
