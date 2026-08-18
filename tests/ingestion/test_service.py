from pathlib import Path

from app.ingestion.models import Document
from app.ingestion.service import IngestionService


def test_ingest_file(tmp_path: Path) -> None:
    file_path = tmp_path / "document.md"
    file_path.write_text("Test document.", encoding="utf-8")

    document = IngestionService().ingest_file(file_path)

    assert document.content == "Test document."
    assert document.metadata["file_name"] == "document.md"


def test_ingest_directory(tmp_path: Path) -> None:
    (tmp_path / "first.md").write_text("First.", encoding="utf-8")
    (tmp_path / "second.txt").write_text("Second.", encoding="utf-8")

    documents = IngestionService().ingest_directory(tmp_path)

    assert len(documents) == 2


def test_chunk_document() -> None:
    document = Document(
        document_id="doc-001",
        source="guide.md",
        content="# Introduction\n\nSome useful information.",
        metadata={"file_type": ".md"},
    )

    chunks = IngestionService().chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-001"
    assert chunks[0].content
