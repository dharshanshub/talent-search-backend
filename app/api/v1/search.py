from __future__ import annotations

import structlog
from fastapi import APIRouter, Request

from app.core.exceptions import BadRequestError, UpstreamServiceError
from app.core.logging import get_correlation_id
from app.schemas.search import SearchRequest, SearchResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search_talent(body: SearchRequest, request: Request) -> SearchResponse:
    """Search for matching candidates using a natural-language query.

    Raises:
        BadRequestError: if the query is empty.
        UpstreamServiceError: if OpenAI or Pinecone calls fail after retries.
    """
    request_id = get_correlation_id()

    if not body.query.strip():
        raise BadRequestError("Search query must not be empty")

    logger.info(
        "search_request",
        query=body.query[:200],
        top_k=body.top_k,
        request_id=request_id,
    )

    # AppException subclasses (BadRequestError, UpstreamServiceError) propagate
    # to the global exception handler registered in main.py — no need to re-wrap.
    service = request.app.state.search_service
    result = await service.search(body, request_id)

    logger.info(
        "search_response",
        candidates_returned=len(result.candidates),
        request_id=request_id,
    )
    return result
