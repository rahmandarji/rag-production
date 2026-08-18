from pydantic import BaseModel, Field


class ClaimVerification(BaseModel):
    claim: str
    grounded: bool
    supporting_chunk_ids: list[str] = Field(default_factory=list)


class GroundingResult(BaseModel):
    grounded: bool
    claims: list[ClaimVerification] = Field(default_factory=list)
    supporting_chunk_ids: list[str] = Field(default_factory=list)
