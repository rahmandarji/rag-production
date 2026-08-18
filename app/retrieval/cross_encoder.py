from sentence_transformers import CrossEncoder

from app.retrieval.models import RetrievalResult
from app.retrieval.reranker import Reranker


class CrossEncoderReranker(Reranker):
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if not results:
            return []

        pairs = [
            (query, result.chunk.content)
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked = [
            result.model_copy(
                update={"reranker_score": float(score)}
            )
            for result, score in zip(results, scores)
        ]

        reranked.sort(
            key=lambda result: result.reranker_score
            if result.reranker_score is not None
            else float("-inf"),
            reverse=True,
        )

        return reranked[:top_k]
