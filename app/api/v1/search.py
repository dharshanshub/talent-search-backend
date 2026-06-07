from __future__ import annotations

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.core.exceptions import BadRequestError
from app.core.logging import get_correlation_id
from app.schemas.search import SearchRequest, SearchResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search_talent(body: SearchRequest, request: Request) -> SearchResponse:
    """Legacy non-streaming search endpoint. Kept for backwards compatibility."""
    request_id = get_correlation_id()

    if not body.query.strip():
        raise BadRequestError("Search query must not be empty")

    logger.info("search_request", query=body.query[:200], top_k=body.top_k, request_id=request_id)

    service = request.app.state.search_service
    result = await service.search(body, request_id)

    logger.info("search_response", candidates_returned=len(result.candidates), request_id=request_id)
    return result


@router.post("/stream")
async def search_stream(body: SearchRequest, request: Request) -> StreamingResponse:
    """Agentic streaming search endpoint.

    Returns a Server-Sent Events stream. Event types:
      data: {"type": "candidates", "data": [...]}   — candidate cards (emitted first if search ran)
      data: {"type": "delta",      "content": "..."}— streamed text token
      data: {"type": "done"}                         — stream complete
      data: {"type": "error",      "message": "..."}— terminal error

    The agent decides whether to query the vector database or respond conversationally,
    maintaining up to 30 turns of conversation history for context.
    """
    request_id = get_correlation_id()

    if not body.query.strip():
        raise BadRequestError("Search query must not be empty")

    logger.info(
        "agent_request",
        query=body.query[:200],
        history_turns=len(body.messages),
        request_id=request_id,
    )

    agent = request.app.state.agent

    async def event_generator():
        async for chunk in agent.run_stream(
            query=body.query,
            history=body.messages,
            top_k=body.top_k,
            request_id=request_id,
        ):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for SSE
        },
    )
