from fastapi import Request

from app.pipeline.rag_pipeline import RAGPipeline


def get_rag_pipeline(request: Request) -> RAGPipeline:
    pipeline = getattr(request.app.state, "rag_pipeline", None)

    if pipeline is None:
        raise RuntimeError("RAG pipeline is not configured.")

    return pipeline
