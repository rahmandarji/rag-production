from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.ingestion.models import Chunk
from app.retrieval.models import RetrievalResult
from app.retrieval.vector_store import VectorStore


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        path: str | Path = "data/qdrant",
        collection_name: str = "documents",
        vector_size: int = 384,
    ) -> None:
        self.path = Path(path)
        self.collection_name = collection_name
        self.vector_size = vector_size

        self.path.mkdir(parents=True, exist_ok=True)

        self.client = QdrantClient(path=str(self.path))

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            collection_info = self.client.get_collection(self.collection_name)

            existing_size = collection_info.config.params.vectors.size

            if existing_size != self.vector_size:
                raise ValueError(
                    f"Collection '{self.collection_name}' has vector size "
                    f"{existing_size}, expected {self.vector_size}."
                )

            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"rag-production:{chunk_id}"))

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        points = []

        for chunk, embedding in zip(chunks, embeddings):
            if len(embedding) != self.vector_size:
                raise ValueError(
                    f"Embedding for {chunk.chunk_id} has dimension "
                    f"{len(embedding)}, expected {self.vector_size}."
                )

            points.append(
                models.PointStruct(
                    id=self._point_id(chunk.chunk_id),
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                    },
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[RetrievalResult]:
        if len(query_embedding) != self.vector_size:
            raise ValueError(
                f"Query embedding has dimension {len(query_embedding)}, "
                f"expected {self.vector_size}."
            )

        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            with_payload=True,
        ).points

        chunks = []

        for result in results:
            payload = result.payload or {}

            chunk = Chunk(
                chunk_id=str(payload["chunk_id"]),
                document_id=str(payload["document_id"]),
                content=str(payload["content"]),
                metadata=dict(payload.get("metadata", {})),
            )

            chunks.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(result.score),
                )
            )

        return chunks
