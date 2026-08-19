from pathlib import Path

from app.core.config import settings
from app.ingestion.chunker import Chunker
from app.ingestion.loader import DocumentLoader
from app.ingestion.models import Chunk, Document


class IngestionService:
    def __init__(
        self,
        loader: DocumentLoader | None = None,
        chunker: Chunker | None = None,
    ) -> None:
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or Chunker()

    def ingest_file(self, path: Path) -> Document:
        return self.loader.load_file(path)

    def ingest_directory(self, directory: Path) -> list[Document]:
        return self.loader.load_directory(directory)

    def chunk_document(self, document: Document) -> list[Chunk]:
        chunks = self.chunker.chunk_document(document)

        if len(chunks) > settings.max_chunks_per_document:
            raise ValueError(
                f"Document '{document.source}' produced "
                f"{len(chunks)} chunks, but the maximum is "
                f"{settings.max_chunks_per_document}."
            )

        if not chunks:
            raise ValueError(
                f"Document produced no chunks: {document.source}"
            )

        return chunks
