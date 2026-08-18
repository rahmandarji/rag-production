from sentence_transformers import SentenceTransformer

from app.ingestion.models import Chunk
from app.retrieval.embeddings import EmbeddingProvider


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, chunks: list[Chunk]) -> list[list[float]]:
        texts = [chunk.content for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        return embedding.tolist()
