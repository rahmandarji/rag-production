from pydantic import BaseModel, Field

from app.ingestion.models import Chunk


class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float = Field(description="Vector similarity score.")
    reranker_score: float | None = Field(
        default=None,
        description="Cross-encoder relevance score.",
    )
