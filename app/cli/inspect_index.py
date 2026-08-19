import json

from app.core.config import settings
from app.retrieval.qdrant_store import QdrantVectorStore


def main() -> None:
    store = QdrantVectorStore(
        path=settings.qdrant_path,
        collection_name=settings.qdrant_collection,
        vector_size=settings.embedding_dimension,
    )

    records, _ = store.client.scroll(
        collection_name=settings.qdrant_collection,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )

    if not records:
        raise SystemExit(
            "No indexed chunks found. Index your documents first."
        )

    for index, record in enumerate(records, start=1):
        payload = record.payload or {}

        print(
            json.dumps(
                {
                    "number": index,
                    "chunk_id": payload.get("chunk_id"),
                    "document_id": payload.get("document_id"),
                    "content": payload.get("content"),
                    "metadata": payload.get("metadata", {}),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print("-" * 80)


if __name__ == "__main__":
    main()
