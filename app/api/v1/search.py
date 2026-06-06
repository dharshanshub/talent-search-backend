from __future__ import annotations

import structlog
from fastapi import APIRouter, Request

from app.core.exceptions import UpstreamServiceError
from app.core.logging import get_correlation_id
from app.schemas.search import SearchRequest, SearchResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search_talent(body: SearchRequest, request: Request) -> SearchResponse:
    """Search for matching candidates using a natural-language query."""
    request_id = get_correlation_id()
    logger.info("search_request", query=body.query, request_id=request_id)

    try:
        service = request.app.state.search_service
        result = await service.search(body, request_id)
    except NotImplementedError:
        raise UpstreamServiceError("search_service", "Search pipeline not yet initialised")

    logger.info(
        "search_response",
        candidates_returned=len(result.candidates),
        request_id=request_id,
    )
    return result
