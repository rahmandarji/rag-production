from app.core.config import Settings, settings
from app.generation.qwen import QwenGenerationProvider
from app.grounding.nli_validator import NLIGroundingValidator
from app.grounding.nli_verifier import NLIVerifier
from app.pipeline.rag_pipeline import RAGPipeline
from app.retrieval.cross_encoder import CrossEncoderReranker
from app.retrieval.embeddings import BGEEmbeddingProvider
from app.retrieval.qdrant_store import QdrantVectorStore
from app.retrieval.retriever import Retriever


def create_rag_pipeline(config: Settings = settings) -> RAGPipeline:
    """Build the production RAG dependency graph from configuration."""

    embedding_provider = BGEEmbeddingProvider(
        model_name=config.embedding_model,
    )

    vector_store = QdrantVectorStore(
        path=config.qdrant_path,
        collection_name=config.qdrant_collection,
        vector_size=config.embedding_dimension,
    )

    reranker = CrossEncoderReranker(
        model_name=config.reranker_model,
    )

    retriever = Retriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        reranker=reranker,
        retrieval_k=config.retrieval_k,
    )

    generator = QwenGenerationProvider(
        model_name=config.generation_model,
        max_new_tokens=config.max_new_tokens,
    )

    verifier = NLIVerifier(
        model_name=config.nli_model,
        threshold=config.nli_threshold,
    )

    grounding_validator = NLIGroundingValidator(
        verifier=verifier,
    )

    return RAGPipeline(
        retriever=retriever,
        generator=generator,
        grounding_validator=grounding_validator,
    )
