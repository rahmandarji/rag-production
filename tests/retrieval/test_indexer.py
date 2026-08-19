from unittest.mock import MagicMock

import pytest

from app.ingestion.models import Chunk
from app.retrieval.indexer import RetrievalIndexer


def make_chunk(chunk_id: str, document_id: str = "doc-1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=f"content-{chunk_id}",
        metadata={},
    )


def test_index_replaces_existing_document() -> None:
    embedding_provider = MagicMock()
    vector_store = MagicMock()

    chunks = [
        make_chunk("chunk-1"),
        make_chunk("chunk-2"),
    ]

    embedding_provider.embed_documents.return_value = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    indexer = RetrievalIndexer(embedding_provider, vector_store)

    indexer.index(chunks)

    vector_store.delete_document.assert_called_once_with("doc-1")
    vector_store.add.assert_called_once_with(
        chunks=chunks,
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )


def test_index_rejects_chunks_from_multiple_documents() -> None:
    embedding_provider = MagicMock()
    vector_store = MagicMock()

    chunks = [
        make_chunk("chunk-1", "doc-1"),
        make_chunk("chunk-2", "doc-2"),
    ]

    indexer = RetrievalIndexer(embedding_provider, vector_store)

    with pytest.raises(ValueError, match="same document"):
        indexer.index(chunks)

    embedding_provider.embed_documents.assert_not_called()
    vector_store.delete_document.assert_not_called()
    vector_store.add.assert_not_called()
