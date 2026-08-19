from pydantic import BaseModel, Field


class EvaluationCase(BaseModel):
    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)

    expected_answer: str | None = None

    expected_chunk_ids: list[str] = Field(default_factory=list)

    should_refuse: bool = False

    answer_keywords: list[str] = Field(default_factory=list)


class CaseEvaluation(BaseModel):
    case_id: str
    retrieval_hit: bool
    retrieval_recall: float
    reciprocal_rank: float

    grounded: bool
    refusal_expected: bool
    refusal_correct: bool

    answer_f1: float
    passed: bool


class EvaluationReport(BaseModel):
    total_cases: int
    retrieval_recall_at_k: float
    mean_reciprocal_rank: float
    grounding_accuracy: float
    refusal_accuracy: float
    answer_f1: float
    overall_pass_rate: float

    cases: list[CaseEvaluation] = Field(default_factory=list)
