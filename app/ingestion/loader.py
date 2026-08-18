from pathlib import Path

from app.ingestion.models import Document


SUPPORTED_EXTENSIONS = {".md", ".txt"}


class DocumentLoader:
    def load_file(self, path: Path) -> Document:
        if not path.is_file():
            raise FileNotFoundError(f"Document not found: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        content = path.read_text(encoding="utf-8")

        if not content.strip():
            raise ValueError(f"Document is empty: {path}")

        return Document(
            document_id=self._document_id(path),
            source=str(path),
            content=content,
            metadata={
                "file_name": path.name,
                "file_type": path.suffix.lower(),
            },
        )

    def load_directory(self, directory: Path) -> list[Document]:
        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")

        documents: list[Document] = []

        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                documents.append(self.load_file(path))

        return documents

    @staticmethod
    def _document_id(path: Path) -> str:
        return str(path.resolve())
