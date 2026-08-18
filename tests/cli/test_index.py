from pathlib import Path
from unittest.mock import MagicMock, patch

from app.cli.index import index_directory


def test_index_directory_indexes_all_document_chunks(
    tmp_path: Path,
) -> None:
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    with (
        patch("app.cli.index.IngestionService") as ingestion_cls,
        patch("app.cli.index.BGEEmbeddingProvider") as embedding_cls,
        patch("app.cli.index.QdrantVectorStore") as vector_store_cls,
        patch("app.cli.index.RetrievalIndexer") as indexer_cls,
    ):
        document = MagicMock()
        document.source = "guide.md"

        chunk_1 = MagicMock()
        chunk_2 = MagicMock()

        ingestion = ingestion_cls.return_value
        ingestion.ingest_directory.return_value = [document]
        ingestion.chunk_document.return_value = [chunk_1, chunk_2]

        total = index_directory(documents_dir)

        ingestion.ingest_directory.assert_called_once_with(documents_dir)
        ingestion.chunk_document.assert_called_once_with(document)

        embedding_cls.assert_called_once()
        vector_store_cls.assert_called_once()
        indexer_cls.assert_called_once()

        indexer_cls.return_value.index.assert_called_once_with(
            [chunk_1, chunk_2]
        )

        assert total == 2
