from unittest.mock import patch

from app.core.config import Settings
from app.core.container import create_rag_pipeline
from app.pipeline.rag_pipeline import RAGPipeline


def test_create_rag_pipeline_wires_configuration() -> None:
    config = Settings(
        embedding_model="test-embedding",
        embedding_dimension=768,
        qdrant_path="test/qdrant",
        qdrant_collection="test-documents",
        reranker_model="test-reranker",
        retrieval_k=10,
        generation_model="test-generation",
        max_new_tokens=128,
        nli_model="test-nli",
        nli_threshold=0.9,
    )

    with (
        patch("app.core.container.BGEEmbeddingProvider") as embedding_cls,
        patch("app.core.container.QdrantVectorStore") as vector_store_cls,
        patch("app.core.container.CrossEncoderReranker") as reranker_cls,
        patch("app.core.container.Retriever") as retriever_cls,
        patch("app.core.container.QwenGenerationProvider") as generator_cls,
        patch("app.core.container.NLIVerifier") as verifier_cls,
        patch("app.core.container.NLIGroundingValidator") as validator_cls,
    ):
        pipeline = create_rag_pipeline(config)

    assert isinstance(pipeline, RAGPipeline)

    embedding_cls.assert_called_once_with(
        model_name="test-embedding",
    )

    vector_store_cls.assert_called_once_with(
        path="test/qdrant",
        collection_name="test-documents",
        vector_size=768,
    )

    reranker_cls.assert_called_once_with(
        model_name="test-reranker",
    )

    retriever_cls.assert_called_once_with(
        embedding_provider=embedding_cls.return_value,
        vector_store=vector_store_cls.return_value,
        reranker=reranker_cls.return_value,
        retrieval_k=10,
    )

    generator_cls.assert_called_once_with(
        model_name="test-generation",
        max_new_tokens=128,
    )

    verifier_cls.assert_called_once_with(
        model_name="test-nli",
        threshold=0.9,
    )

    validator_cls.assert_called_once_with(
        verifier=verifier_cls.return_value,
    )
