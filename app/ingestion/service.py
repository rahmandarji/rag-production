from pathlib import Path

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
        return self.chunker.chunk_document(document)
