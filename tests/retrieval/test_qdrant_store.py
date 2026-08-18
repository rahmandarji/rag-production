from pathlib import Path

import pytest

from app.ingestion.models import Chunk
from app.retrieval.qdrant_store import QdrantVectorStore


def make_chunk(
    chunk_id: str,
    content: str,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        content=content,
        metadata={
            "file_name": "test.md",
            "file_type": ".md",
        },
    )


def test_collection_is_created(tmp_path: Path) -> None:
    store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="test_collection",
        vector_size=2,
    )

    assert store.client.collection_exists("test_collection")


def test_add_and_search(tmp_path: Path) -> None:
    store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="test_collection",
        vector_size=2,
    )

    chunks = [
        make_chunk("chunk-1", "Path parameters are declared in routes."),
        make_chunk("chunk-2", "Query parameters are optional."),
    ]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    store.add(chunks, embeddings)

    results = store.search([1.0, 0.0], limit=1)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk-1"
    assert results[0].chunk.content == "Path parameters are declared in routes."


def test_add_requires_matching_lengths(tmp_path: Path) -> None:
    store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="test_collection",
        vector_size=2,
    )

    chunks = [make_chunk("chunk-1", "Test.")]

    with pytest.raises(ValueError):
        store.add(chunks, [])


def test_rejects_wrong_embedding_dimension(tmp_path: Path) -> None:
    store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="test_collection",
        vector_size=2,
    )

    chunks = [make_chunk("chunk-1", "Test.")]

    with pytest.raises(ValueError):
        store.add(chunks, [[1.0, 2.0, 3.0]])


def test_rejects_wrong_query_dimension(tmp_path: Path) -> None:
    store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="test_collection",
        vector_size=2,
    )

    with pytest.raises(ValueError):
        store.search([1.0, 2.0, 3.0])


def test_rejects_invalid_limit(tmp_path: Path) -> None:
    store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="test_collection",
        vector_size=2,
    )

    with pytest.raises(ValueError):
        store.search([1.0, 0.0], limit=0)
