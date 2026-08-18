from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    retrieval_limit: int = Field(default=5, ge=1)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")

        return value


class SourceResponse(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, str]


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[SourceResponse] = Field(default_factory=list)
