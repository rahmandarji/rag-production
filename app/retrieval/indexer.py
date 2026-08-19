from app.ingestion.models import Chunk
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.vector_store import VectorStore


class RetrievalIndexer:
    """Index document chunks with idempotent document replacement."""

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

        document_ids = {chunk.document_id for chunk in chunks}

        if len(document_ids) != 1:
            raise ValueError(
                "All chunks in one indexing operation must belong to "
                "the same document."
            )

        document_id = next(iter(document_ids))

        embeddings = self.embedding_provider.embed_documents(chunks)

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings must match number of chunks."
            )

        # Replace the document atomically from the application's perspective:
        # remove old chunks first, then upsert the current chunks.
        self.vector_store.delete_document(document_id)

        self.vector_store.add(
            chunks=chunks,
            embeddings=embeddings,
        )
