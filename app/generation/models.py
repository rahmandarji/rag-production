from pydantic import BaseModel, Field


class AnswerSource(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, str] = Field(default_factory=dict)


class GeneratedAnswer(BaseModel):
    answer: str
    sources: list[AnswerSource] = Field(default_factory=list)
