import math

from app.ingestion.models import Chunk
from app.retrieval.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)


def test_query_embedding_dimension() -> None:
    provider = SentenceTransformerEmbeddingProvider()

    embedding = provider.embed_query("How do path parameters work?")

    assert len(embedding) == 384


def test_query_embedding_is_normalized() -> None:
    provider = SentenceTransformerEmbeddingProvider()

    embedding = provider.embed_query("How do path parameters work?")

    norm = math.sqrt(sum(value * value for value in embedding))

    assert math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5)


def test_document_embedding_dimension() -> None:
    provider = SentenceTransformerEmbeddingProvider()

    chunks = [
        Chunk(
            chunk_id="doc-001:chunk-0000",
            document_id="doc-001",
            content="Path parameters are declared in the route.",
        ),
        Chunk(
            chunk_id="doc-001:chunk-0001",
            document_id="doc-001",
            content="Query parameters are optional.",
        ),
    ]

    embeddings = provider.embed_documents(chunks)

    assert len(embeddings) == 2
    assert all(len(embedding) == 384 for embedding in embeddings)
