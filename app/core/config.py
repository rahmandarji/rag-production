from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "rag-production"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = Field(default=384, gt=0)

    qdrant_path: str = "data/qdrant"
    qdrant_collection: str = "documents"

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    retrieval_k: int = Field(default=20, gt=0)
    max_retrieval_limit: int = Field(default=20, gt=0)

    generation_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = Field(default=256, gt=0)

    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    nli_threshold: float = Field(default=0.8, gt=0, le=1)

    max_document_size_bytes: int = Field(
        default=10 * 1024 * 1024,
        gt=0,
    )

    max_documents_per_batch: int = Field(
        default=100,
        gt=0,
    )

    max_chunks_per_document: int = Field(
        default=10_000,
        gt=0,
    )

    max_query_length: int = Field(
        default=2000,
        gt=0,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        extra="ignore",
    )

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        value = value.strip().upper()

        allowed = {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        if value not in allowed:
            raise ValueError(
                f"log_level must be one of: {', '.join(sorted(allowed))}"
            )

        return value

    @model_validator(mode="after")
    def validate_limits(self) -> "Settings":
        if self.max_retrieval_limit > self.retrieval_k:
            raise ValueError(
                "max_retrieval_limit cannot exceed retrieval_k."
            )

        return self


settings = Settings()
