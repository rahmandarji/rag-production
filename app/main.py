from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import settings
from app.core.container import create_rag_pipeline
from app.core.logging import configure_logging


configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_pipeline = create_rag_pipeline()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(api_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "environment": settings.environment,
    }
