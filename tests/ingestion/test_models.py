import pytest
from pydantic import ValidationError

from app.ingestion.models import Chunk, Document


def test_document_creation() -> None:
    document = Document(
        document_id="doc-001",
        source="example.md",
        content="Hello world.",
        metadata={"file_type": "markdown"},
    )

    assert document.document_id == "doc-001"
    assert document.source == "example.md"
    assert document.content == "Hello world."
    assert document.metadata["file_type"] == "markdown"


def test_document_metadata_defaults_to_empty_dict() -> None:
    document = Document(
        document_id="doc-001",
        source="example.txt",
        content="Hello world.",
    )

    assert document.metadata == {}


@pytest.mark.parametrize(
    "field",
    ["document_id", "source", "content"],
)
def test_required_text_fields_cannot_be_empty(field: str) -> None:
    values = {
        "document_id": "doc-001",
        "source": "example.txt",
        "content": "Hello world.",
    }
    values[field] = ""

    with pytest.raises(ValidationError):
        Document(**values)


def test_chunk_creation() -> None:
    chunk = Chunk(
        chunk_id="doc-001:chunk-001",
        document_id="doc-001",
        content="Chunk content.",
        metadata={"section": "Introduction"},
    )

    assert chunk.chunk_id == "doc-001:chunk-001"
    assert chunk.document_id == "doc-001"
    assert chunk.content == "Chunk content."
    assert chunk.metadata["section"] == "Introduction"


def test_chunk_metadata_defaults_to_empty_dict() -> None:
    chunk = Chunk(
        chunk_id="doc-001:chunk-001",
        document_id="doc-001",
        content="Chunk content.",
    )

    assert chunk.metadata == {}
