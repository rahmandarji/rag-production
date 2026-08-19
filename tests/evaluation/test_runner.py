from unittest.mock import Mock

from app.evaluation.models import EvaluationCase
from app.evaluation.runner import RAGEvaluator
from app.generation.models import AnswerSource, GeneratedAnswer
from app.grounding.models import GroundingResult
from app.pipeline.rag_pipeline import RAGResponse
from app.retrieval.models import RetrievalResult
from app.ingestion.models import Chunk


def make_result(
    answer: str,
    grounded: bool,
) -> RAGResponse:
    chunk = Chunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="Path parameters are declared in the route path.",
        metadata={},
    )

    retrieval = RetrievalResult(
        chunk=chunk,
        score=0.95,
    )

    return RAGResponse(
        answer=GeneratedAnswer(
            answer=answer,
            sources=[
                AnswerSource(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    content=chunk.content,
                    metadata={},
                )
            ],
        ),
        grounding=GroundingResult(
            grounded=grounded,
            claims=[],
            supporting_chunk_ids=(
                ["chunk-1"] if grounded else []
            ),
        ),
        evidence=[retrieval],
    )


def test_evaluator_accepts_grounded_answer() -> None:
    pipeline = Mock()
    pipeline.query.return_value = make_result(
        "Path parameters are declared in the route path.",
        True,
    )

    evaluator = RAGEvaluator(
        pipeline=pipeline,
        retrieval_limit=5,
    )

    case = EvaluationCase(
        case_id="case-1",
        question="How are path parameters declared?",
        expected_answer=(
            "Path parameters are declared in the route path."
        ),
        expected_chunk_ids=["chunk-1"],
    )

    result = evaluator.evaluate_case(case)

    assert result.retrieval_hit is True
    assert result.retrieval_recall == 1.0
    assert result.reciprocal_rank == 1.0
    assert result.grounded is True
    assert result.answer_f1 == 1.0
    assert result.passed is True


def test_evaluator_accepts_correct_refusal() -> None:
    pipeline = Mock()
    pipeline.query.return_value = make_result(
        "I don't have enough information in the provided documents to answer this question.",
        False,
    )

    evaluator = RAGEvaluator(pipeline)

    case = EvaluationCase(
        case_id="refusal-1",
        question="What is the capital of France?",
        should_refuse=True,
    )

    result = evaluator.evaluate_case(case)

    assert result.refusal_correct is True
    assert result.passed is True


def test_evaluator_rejects_incorrect_refusal() -> None:
    pipeline = Mock()
    pipeline.query.return_value = make_result(
        "Paris is the capital of France.",
        True,
    )

    evaluator = RAGEvaluator(pipeline)

    case = EvaluationCase(
        case_id="refusal-2",
        question="What is the capital of France?",
        should_refuse=True,
    )

    result = evaluator.evaluate_case(case)

    assert result.refusal_correct is False
    assert result.passed is False


def test_evaluator_report() -> None:
    pipeline = Mock()
    pipeline.query.return_value = make_result(
        "Path parameters are declared in the route path.",
        True,
    )

    evaluator = RAGEvaluator(pipeline)

    cases = [
        EvaluationCase(
            case_id="case-1",
            question="How are path parameters declared?",
            expected_answer=(
                "Path parameters are declared in the route path."
            ),
            expected_chunk_ids=["chunk-1"],
        ),
        EvaluationCase(
            case_id="case-2",
            question="How are path parameters declared?",
            expected_answer=(
                "Path parameters are declared in the route path."
            ),
            expected_chunk_ids=["chunk-1"],
        ),
    ]

    report = evaluator.evaluate(cases)

    assert report.total_cases == 2
    assert report.retrieval_recall_at_k == 1.0
    assert report.mean_reciprocal_rank == 1.0
    assert report.grounding_accuracy == 1.0
    assert report.answer_f1 == 1.0
    assert report.overall_pass_rate == 1.0
