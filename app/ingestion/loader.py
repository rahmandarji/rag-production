from pathlib import Path

from app.core.config import settings
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

        file_size = path.stat().st_size

        if file_size > settings.max_document_size_bytes:
            raise ValueError(
                f"Document exceeds maximum size of "
                f"{settings.max_document_size_bytes} bytes: {path}"
            )

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Document is not valid UTF-8: {path}"
            ) from exc
        except OSError as exc:
            raise OSError(
                f"Failed to read document: {path}"
            ) from exc

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
            raise NotADirectoryError(
                f"Directory not found: {directory}"
            )

        paths = [
            path
            for path in sorted(directory.rglob("*"))
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if len(paths) > settings.max_documents_per_batch:
            raise ValueError(
                f"Directory contains {len(paths)} supported documents, "
                f"but the maximum is "
                f"{settings.max_documents_per_batch}."
            )

        documents: list[Document] = []

        for path in paths:
            try:
                document = self.load_file(path)
            except (ValueError, OSError) as exc:
                raise ValueError(
                    f"Failed to ingest document '{path}': {exc}"
                ) from exc

            documents.append(document)

        return documents

    @staticmethod
    def _document_id(path: Path) -> str:
        return str(path.resolve())
