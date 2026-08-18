from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.models import RetrievalResult
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
        retrieval_k: int = 20,
    ) -> None:
        if retrieval_k <= 0:
            raise ValueError("retrieval_k must be greater than 0.")

        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.reranker = reranker
        self.retrieval_k = retrieval_k

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        query_embedding = self.embedding_provider.embed_query(query)

        candidates = self.vector_store.search(
            query_embedding=query_embedding,
            limit=max(limit, self.retrieval_k),
        )

        if self.reranker is None:
            return candidates[:limit]

        return self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=limit,
        )
