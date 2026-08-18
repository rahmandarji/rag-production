from pathlib import Path

from app.core.config import settings
from app.ingestion.service import IngestionService
from app.retrieval.embeddings import BGEEmbeddingProvider
from app.retrieval.indexer import RetrievalIndexer
from app.retrieval.qdrant_store import QdrantVectorStore


def index_directory(directory: Path) -> int:
    ingestion = IngestionService()

    documents = ingestion.ingest_directory(directory)

    embedding_provider = BGEEmbeddingProvider(
        model_name=settings.embedding_model,
    )

    vector_store = QdrantVectorStore(
        path=settings.qdrant_path,
        collection_name=settings.qdrant_collection,
        vector_size=settings.embedding_dimension,
    )

    indexer = RetrievalIndexer(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    total_chunks = 0

    for document in documents:
        chunks = ingestion.chunk_document(document)
        indexer.index(chunks)
        total_chunks += len(chunks)

        print(
            f"Indexed {len(chunks)} chunks from "
            f"{document.source}"
        )

    return total_chunks


def main() -> None:
    directory = Path("data/documents")

    if not directory.exists():
        raise SystemExit(
            f"Document directory does not exist: {directory}"
        )

    total_chunks = index_directory(directory)

    print(f"Indexed {total_chunks} chunks total.")


if __name__ == "__main__":
    main()
