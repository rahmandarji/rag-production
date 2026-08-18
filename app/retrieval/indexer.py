from app.ingestion.models import Chunk
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.vector_store import VectorStore


class RetrievalIndexer:
    """Index document chunks into the vector store."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        embeddings = self.embedding_provider.embed_documents(chunks)

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings must match number of chunks."
            )

        self.vector_store.add(
            chunks=chunks,
            embeddings=embeddings,
        )
