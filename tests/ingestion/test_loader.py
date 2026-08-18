from pathlib import Path

import pytest

from app.ingestion.loader import DocumentLoader


def test_load_markdown_file(tmp_path: Path) -> None:
    file_path = tmp_path / "example.md"
    file_path.write_text("# Hello\n\nThis is a document.", encoding="utf-8")

    document = DocumentLoader().load_file(file_path)

    assert document.source == str(file_path)
    assert document.content == "# Hello\n\nThis is a document."
    assert document.metadata["file_name"] == "example.md"
    assert document.metadata["file_type"] == ".md"


def test_load_text_file(tmp_path: Path) -> None:
    file_path = tmp_path / "example.txt"
    file_path.write_text("Plain text document.", encoding="utf-8")

    document = DocumentLoader().load_file(file_path)

    assert document.content == "Plain text document."
    assert document.metadata["file_type"] == ".txt"


def test_reject_unsupported_file_type(tmp_path: Path) -> None:
    file_path = tmp_path / "example.pdf"
    file_path.write_text("PDF placeholder.", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        DocumentLoader().load_file(file_path)


def test_reject_empty_file(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.md"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Document is empty"):
        DocumentLoader().load_file(file_path)


def test_reject_missing_file(tmp_path: Path) -> None:
    file_path = tmp_path / "missing.md"

    with pytest.raises(FileNotFoundError):
        DocumentLoader().load_file(file_path)


def test_load_directory_recursively(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()

    (tmp_path / "first.md").write_text("First document.", encoding="utf-8")
    (nested / "second.txt").write_text("Second document.", encoding="utf-8")
    (tmp_path / "ignored.pdf").write_text("Ignored.", encoding="utf-8")

    documents = DocumentLoader().load_directory(tmp_path)

    assert len(documents) == 2
    assert [document.metadata["file_name"] for document in documents] == [
        "first.md",
        "second.txt",
    ]


def test_load_directory_returns_empty_list_when_no_supported_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "ignored.pdf").write_text("Ignored.", encoding="utf-8")

    documents = DocumentLoader().load_directory(tmp_path)

    assert documents == []


def test_load_directory_rejects_missing_directory(tmp_path: Path) -> None:
    missing_directory = tmp_path / "missing"

    with pytest.raises(NotADirectoryError):
        DocumentLoader().load_directory(missing_directory)
