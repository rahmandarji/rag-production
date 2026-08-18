import pytest

from app.ingestion.chunker import Chunker
from app.ingestion.models import Document


def test_chunk_markdown_document() -> None:
    document = Document(
        document_id="doc-001",
        source="guide.md",
        content=(
            "# Introduction\n\n"
            "This is the introduction.\n\n"
            "## Authentication\n\n"
            "Authentication allows users to access the API.\n"
        ),
        metadata={
            "file_name": "guide.md",
            "file_type": ".md",
        },
    )

    chunks = Chunker(chunk_size=100, chunk_overlap=20).chunk_document(document)

    assert len(chunks) >= 2
    assert all(chunk.document_id == "doc-001" for chunk in chunks)
    assert all(chunk.content.strip() for chunk in chunks)
    assert all(chunk.chunk_id for chunk in chunks)


def test_markdown_headers_are_preserved_as_metadata() -> None:
    document = Document(
        document_id="doc-001",
        source="guide.md",
        content="# Introduction\n\nSome useful information.",
        metadata={"file_type": ".md"},
    )

    chunks = Chunker().chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].metadata["h1"] == "Introduction"


def test_chunk_text_document() -> None:
    document = Document(
        document_id="doc-001",
        source="notes.txt",
        content="A " * 1000,
        metadata={
            "file_name": "notes.txt",
            "file_type": ".txt",
        },
    )

    chunks = Chunker(chunk_size=100, chunk_overlap=20).chunk_document(document)

    assert len(chunks) > 1
    assert all(chunk.document_id == "doc-001" for chunk in chunks)


def test_chunk_ids_are_unique() -> None:
    document = Document(
        document_id="doc-001",
        source="notes.txt",
        content="A " * 1000,
        metadata={"file_type": ".txt"},
    )

    chunks = Chunker(chunk_size=100, chunk_overlap=20).chunk_document(document)

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_invalid_chunk_configuration() -> None:
    with pytest.raises(ValueError):
        Chunker(chunk_size=0)

    with pytest.raises(ValueError):
        Chunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError):
        Chunker(chunk_size=100, chunk_overlap=-1)
