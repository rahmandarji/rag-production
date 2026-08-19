from pydantic import BaseModel, Field

from app.core.metrics import metrics
from app.generation.models import AnswerSource, GeneratedAnswer
from app.generation.provider import GenerationProvider
from app.grounding.models import GroundingResult
from app.grounding.validator import GroundingValidator
from app.retrieval.models import RetrievalResult
from app.retrieval.retriever import Retriever


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I don't have enough information in the provided documents to answer this question."
)


class RAGResponse(BaseModel):
    answer: GeneratedAnswer
    grounding: GroundingResult
    evidence: list[RetrievalResult] = Field(default_factory=list)


class RAGPipeline:
    """
    Closed-world RAG orchestration.

    Flow:
        query
          -> retrieval
          -> generation using retrieved evidence only
          -> grounding verification
          -> final answer + supporting sources

    If retrieval or grounding is insufficient, the pipeline refuses
    instead of allowing unsupported model knowledge into the answer.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: GenerationProvider,
        grounding_validator: GroundingValidator,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.grounding_validator = grounding_validator

    def query(
        self,
        query: str,
        retrieval_limit: int = 5,
    ) -> RAGResponse:
        query = query.strip()

        if not query:
            raise ValueError("Query must not be empty.")

        if retrieval_limit <= 0:
            raise ValueError("retrieval_limit must be greater than 0.")

        metrics.increment("rag_queries_total")

        evidence = self.retriever.search(
            query=query,
            limit=retrieval_limit,
        )

        if not evidence:
            metrics.increment("retrieval_empty_total")
            metrics.increment("rag_refusals_total")

            return self._refusal(evidence=[])

        generated_answer = self.generator.generate(
            query=query,
            evidence=evidence,
        )

        grounding = self.grounding_validator.validate(
            answer=generated_answer,
            evidence=evidence,
        )

        if not grounding.grounded:
            metrics.increment("rag_refusals_total")

            return RAGResponse(
                answer=GeneratedAnswer(
                    answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                    sources=[],
                ),
                grounding=grounding,
                evidence=evidence,
            )

        metrics.increment("rag_grounded_total")

        sources = self._build_sources(
            evidence=evidence,
            supporting_chunk_ids=grounding.supporting_chunk_ids,
        )

        final_answer = GeneratedAnswer(
            answer=generated_answer.answer,
            sources=sources,
        )

        return RAGResponse(
            answer=final_answer,
            grounding=grounding,
            evidence=evidence,
        )

    @staticmethod
    def _refusal(
        evidence: list[RetrievalResult],
    ) -> RAGResponse:
        return RAGResponse(
            answer=GeneratedAnswer(
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                sources=[],
            ),
            grounding=GroundingResult(
                grounded=False,
                claims=[],
                supporting_chunk_ids=[],
            ),
            evidence=evidence,
        )

    @staticmethod
    def _build_sources(
        evidence: list[RetrievalResult],
        supporting_chunk_ids: list[str],
    ) -> list[AnswerSource]:
        supporting_ids = set(supporting_chunk_ids)

        return [
            AnswerSource(
                chunk_id=result.chunk.chunk_id,
                document_id=result.chunk.document_id,
                content=result.chunk.content,
                metadata=result.chunk.metadata,
            )
            for result in evidence
            if result.chunk.chunk_id in supporting_ids
        ]
