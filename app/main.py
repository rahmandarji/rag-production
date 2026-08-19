from contextlib import asynccontextmanager
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.exceptions import internal_exception_handler
from app.api.routes import router as api_router
from app.core.config import settings
from app.core.container import create_rag_pipeline
from app.core.logging import configure_logging


configure_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_pipeline = create_rag_pipeline()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

app.add_exception_handler(Exception, internal_exception_handler)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id

    start = perf_counter()

    logger.info(
        "request_started",
        extra={
            "event": "request_started",
            "request_id": request_id,
            "method": request.method,
            "endpoint": request.url.path,
        },
    )

    response = await call_next(request)

    duration_ms = round((perf_counter() - start) * 1000, 2)

    logger.info(
        "request_completed",
        extra={
            "event": "request_completed",
            "request_id": request_id,
            "method": request.method,
            "endpoint": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
        },
    )

    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(api_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "environment": settings.environment,
    }
