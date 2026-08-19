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


def test_reindexing_same_chunks_does_not_duplicate_points(
    tmp_path: Path,
) -> None:
    store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="test_collection",
        vector_size=2,
    )

    chunks = [
        make_chunk("chunk-1", "First."),
        make_chunk("chunk-2", "Second."),
    ]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    store.add(chunks, embeddings)
    store.add(chunks, embeddings)

    records, _ = store.client.scroll(
        collection_name="test_collection",
        limit=100,
        with_payload=True,
        with_vectors=False,
    )

    assert len(records) == 2


def test_reindexing_changed_document_removes_stale_chunks(
    tmp_path: Path,
) -> None:
    store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="test_collection",
        vector_size=2,
    )

    original_chunks = [
        make_chunk("chunk-1", "First."),
        make_chunk("chunk-2", "Second."),
        make_chunk("chunk-3", "Third."),
    ]

    store.add(
        original_chunks,
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ],
    )

    updated_chunks = [
        make_chunk("chunk-1", "First updated."),
        make_chunk("chunk-2", "Second updated."),
    ]

    store.add(
        updated_chunks,
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
    )

    records, _ = store.client.scroll(
        collection_name="test_collection",
        limit=100,
        with_payload=True,
        with_vectors=False,
    )

    chunk_ids = {
        record.payload["chunk_id"]
        for record in records
    }

    assert chunk_ids == {"chunk-1", "chunk-2"}


def test_rejects_duplicate_chunk_ids(
    tmp_path: Path,
) -> None:
    store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="test_collection",
        vector_size=2,
    )

    chunks = [
        make_chunk("chunk-1", "First."),
        make_chunk("chunk-1", "Duplicate."),
    ]

    with pytest.raises(ValueError, match="Duplicate chunk IDs"):
        store.add(
            chunks,
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )


def test_rejects_chunks_from_multiple_documents(
    tmp_path: Path,
) -> None:
    store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="test_collection",
        vector_size=2,
    )

    chunk_1 = make_chunk("chunk-1", "First.")
    chunk_2 = make_chunk("chunk-2", "Second.")
    chunk_2.document_id = "doc-002"

    with pytest.raises(ValueError, match="same document"):
        store.add(
            [chunk_1, chunk_2],
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        )


def test_reindexing_same_document_does_not_duplicate_chunks(
    tmp_path: Path,
) -> None:
    store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="test_collection",
        vector_size=2,
    )

    chunks = [
        make_chunk("chunk-1", "First."),
        make_chunk("chunk-2", "Second."),
    ]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    store.add(chunks, embeddings)
    store.add(chunks, embeddings)

    info = store.client.get_collection("test_collection")

    assert info.points_count == 2


def test_delete_document_removes_all_document_chunks(
    tmp_path: Path,
) -> None:
    store = QdrantVectorStore(
        path=tmp_path / "qdrant",
        collection_name="test_collection",
        vector_size=2,
    )

    document_one_chunks = [
        make_chunk("chunk-1", "First."),
        make_chunk("chunk-2", "Second."),
    ]

    document_two_chunk = make_chunk("chunk-3", "Other.")
    document_two_chunk.document_id = "doc-002"

    store.add(
        document_one_chunks,
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
    )

    store.add(
        [document_two_chunk],
        [
            [0.7, 0.7],
        ],
    )

    store.delete_document("doc-001")

    info = store.client.get_collection("test_collection")

    assert info.points_count == 1

    results = store.search([0.7, 0.7], limit=5)

    assert len(results) == 1
    assert results[0].chunk.document_id == "doc-002"
