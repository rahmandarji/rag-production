from fastapi import APIRouter, Depends

from app.api.dependencies import get_rag_pipeline
from app.api.schemas import QueryRequest, QueryResponse, SourceResponse
from app.pipeline.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/api/v1", tags=["rag"])


@router.post("/query", response_model=QueryResponse)
def query_rag(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    result = pipeline.query(
        query=request.query,
        retrieval_limit=request.retrieval_limit,
    )

    sources = [
        SourceResponse(
            chunk_id=source.chunk_id,
            document_id=source.document_id,
            content=source.content,
            metadata=source.metadata,
        )
        for source in result.answer.sources
    ]

    return QueryResponse(
        answer=result.answer.answer,
        grounded=result.grounding.grounded,
        sources=sources,
    )
