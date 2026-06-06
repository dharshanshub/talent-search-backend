from __future__ import annotations

import traceback

import structlog
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import AppException
from app.core.logging import get_correlation_id
from app.schemas.errors import ErrorResponse

logger = structlog.get_logger(__name__)


def _error_response(
    request: Request,
    status_code: int,
    error_code: str,
    message: str,
) -> JSONResponse:
    body = ErrorResponse(
        error_code=error_code,
        message=message,
        path=str(request.url.path),
        method=request.method,
        request_id=get_correlation_id(),
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    logger.warning(
        "application_error",
        error_code=exc.error_code,
        message=exc.message,
        status_code=exc.status_code,
        path=request.url.path,
        method=request.method,
    )
    return _error_response(request, exc.status_code, exc.error_code, exc.message)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = exc.errors()
    detail = "; ".join(
        f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in errors
    )
    logger.info(
        "validation_error",
        detail=detail,
        path=request.url.path,
        method=request.method,
    )
    return _error_response(request, 422, "VALIDATION_ERROR", detail)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled_exception",
        exc_type=type(exc).__name__,
        path=request.url.path,
        method=request.method,
        traceback=traceback.format_exc(),
    )
    return _error_response(
        request,
        500,
        "INTERNAL_ERROR",
        "An internal server error occurred. Please try again later.",
    )
