from fastapi import FastAPI

from app.core.config import settings
from app.core.logging import configure_logging


configure_logging()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "environment": settings.environment,
    }
