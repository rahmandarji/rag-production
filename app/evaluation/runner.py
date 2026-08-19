from app.evaluation.metrics import (
    reciprocal_rank,
    retrieval_recall,
    token_f1,
)
from app.evaluation.models import (
    CaseEvaluation,
    EvaluationCase,
    EvaluationReport,
)
from app.pipeline.rag_pipeline import RAGPipeline


def _is_refusal(answer: str) -> bool:
    normalized = answer.lower()

    refusal_phrases = (
        "don't have enough information",
        "do not have enough information",
        "insufficient information",
    )

    return any(
        phrase in normalized
        for phrase in refusal_phrases
    )


class RAGEvaluator:
    def __init__(
        self,
        pipeline: RAGPipeline,
        retrieval_limit: int = 5,
    ) -> None:
        if retrieval_limit <= 0:
            raise ValueError(
                "retrieval_limit must be greater than 0."
            )

        self.pipeline = pipeline
        self.retrieval_limit = retrieval_limit

    def evaluate_case(
        self,
        case: EvaluationCase,
    ) -> CaseEvaluation:
        response = self.pipeline.query(
            query=case.question,
            retrieval_limit=self.retrieval_limit,
        )

        retrieved_ids = [
            result.chunk.chunk_id
            for result in response.evidence
        ]

        recall = retrieval_recall(
            retrieved_ids,
            case.expected_chunk_ids,
        )

        rank = reciprocal_rank(
            retrieved_ids,
            case.expected_chunk_ids,
        )

        retrieval_hit = (
            recall > 0
            if case.expected_chunk_ids
            else True
        )

        grounded = response.grounding.grounded

        refusal = _is_refusal(
            response.answer.answer
        )

        refusal_correct = (
            refusal == case.should_refuse
        )

        if case.should_refuse:
            answer_score = (
                1.0 if refusal else 0.0
            )
        elif case.expected_answer:
            answer_score = token_f1(
                response.answer.answer,
                case.expected_answer,
            )
        elif case.answer_keywords:
            answer_tokens = set(
                case.answer_keywords
            )
            response_tokens = set(
                response.answer.answer.lower().split()
            )

            matched = sum(
                keyword.lower() in response_tokens
                for keyword in answer_tokens
            )

            answer_score = (
                matched / len(answer_tokens)
                if answer_tokens
                else 1.0
            )
        else:
            answer_score = 1.0

        if case.should_refuse:
            passed = refusal_correct
        else:
            passed = (
                retrieval_hit
                and grounded
                and not refusal
                and answer_score >= 0.5
            )

        return CaseEvaluation(
            case_id=case.case_id,
            retrieval_hit=retrieval_hit,
            retrieval_recall=recall,
            reciprocal_rank=rank,
            grounded=grounded,
            refusal_expected=case.should_refuse,
            refusal_correct=refusal_correct,
            answer_f1=answer_score,
            passed=passed,
        )

    def evaluate(
        self,
        cases: list[EvaluationCase],
    ) -> EvaluationReport:
        if not cases:
            raise ValueError(
                "Evaluation dataset must not be empty."
            )

        results = [
            self.evaluate_case(case)
            for case in cases
        ]

        total = len(results)

        return EvaluationReport(
            total_cases=total,
            retrieval_recall_at_k=(
                sum(
                    result.retrieval_recall
                    for result in results
                )
                / total
            ),
            mean_reciprocal_rank=(
                sum(
                    result.reciprocal_rank
                    for result in results
                )
                / total
            ),
            grounding_accuracy=(
                sum(
                    result.grounded
                    == (
                        not result.refusal_expected
                    )
                    for result in results
                )
                / total
            ),
            refusal_accuracy=(
                sum(
                    result.refusal_correct
                    for result in results
                )
                / total
            ),
            answer_f1=(
                sum(
                    result.answer_f1
                    for result in results
                )
                / total
            ),
            overall_pass_rate=(
                sum(
                    result.passed
                    for result in results
                )
                / total
            ),
            cases=results,
        )
