import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def internal_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception(
        "Unhandled API exception",
        extra={
            "event": "internal_exception",
            "method": request.method,
            "endpoint": request.url.path,
            "request_id": getattr(request.state, "request_id", None),
        },
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
        headers={
            "X-Request-ID": getattr(request.state, "request_id", ""),
        },
    )
