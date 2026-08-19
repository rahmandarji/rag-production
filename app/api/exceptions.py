import logging

from fastapi import Request
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


async def internal_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    logger.exception(
        "internal_server_error",
        extra={
            "event": "internal_server_error",
            "request_id": request_id,
            "method": request.method,
            "endpoint": request.url.path,
        },
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred.",
        },
    )
