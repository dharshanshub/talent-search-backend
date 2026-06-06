from __future__ import annotations

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def liveness() -> dict:
    return {"status": "ok"}


@router.get("/readyz")
async def readiness(request: Request) -> JSONResponse:
    """Check connectivity to Pinecone and OpenAI. Returns 503 if either is unreachable."""
    checks: dict[str, str] = {}
    healthy = True

    # Pinecone ping
    try:
        store = request.app.state.vector_store
        await store.ping()
        checks["pinecone"] = "ok"
    except NotImplementedError:
        # Still a stub — treat as passing during Phase 1
        checks["pinecone"] = "stub"
    except Exception as exc:
        logger.warning("readyz_pinecone_failed", error=str(exc))
        checks["pinecone"] = "unreachable"
        healthy = False

    # OpenAI ping (lightweight: list models)
    try:
        openai_client = request.app.state.openai_client
        await openai_client.models.list()
        checks["openai"] = "ok"
    except Exception as exc:
        logger.warning("readyz_openai_failed", error=str(exc))
        checks["openai"] = "unreachable"
        healthy = False

    status_code = 200 if healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ok" if healthy else "degraded", "checks": checks},
    )
