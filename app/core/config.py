from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "rag-production"
    environment: str = "development"
    log_level: str = "INFO"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384

    qdrant_path: str = "data/qdrant"
    qdrant_collection: str = "documents"

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    retrieval_k: int = 20

    generation_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 256

    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    nli_threshold: float = 0.8

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
