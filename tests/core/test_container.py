from unittest.mock import patch

from app.core.config import Settings
from app.core.container import create_rag_pipeline


def test_create_rag_pipeline_wires_configuration() -> None:
    config = Settings(
        embedding_model="test-embedding",
        embedding_dimension=384,
        qdrant_path="data/test-qdrant",
        qdrant_collection="test-documents",
        reranker_model="test-reranker",
        retrieval_k=20,
        max_retrieval_limit=20,
        generation_model="test-generation",
        max_new_tokens=128,
        nli_model="test-nli",
        nli_threshold=0.9,
    )

    with (
        patch("app.core.container.BGEEmbeddingProvider"),
        patch("app.core.container.QdrantVectorStore"),
        patch("app.core.container.CrossEncoderReranker"),
        patch("app.core.container.QwenGenerationProvider"),
        patch("app.core.container.NLIVerifier"),
    ):
        pipeline = create_rag_pipeline(config)

    assert pipeline is not None
