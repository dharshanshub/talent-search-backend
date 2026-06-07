from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import set_correlation_id

logger = structlog.get_logger(__name__)

CORRELATION_HEADER = "X-Request-ID"

# Probe paths that are infrastructure noise — never log these
SILENT_PATHS = {"/healthz", "/health", "/ping"}


class CorrelationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(CORRELATION_HEADER) or str(uuid.uuid4())
        set_correlation_id(request_id)

        silent = request.url.path in SILENT_PATHS

        start = time.perf_counter()
        if not silent:
            logger.info(
                "request_started",
                method=request.method,
                path=request.url.path,
            )

        response = await call_next(request)

        if not silent:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "request_finished",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )

        response.headers[CORRELATION_HEADER] = request_id
        return response
