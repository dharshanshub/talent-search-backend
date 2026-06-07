from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.v1.auth import router as auth_router
from app.api.v1.candidates import router as candidates_router
from app.api.v1.knowledge_base import router as knowledge_base_router
from app.api.v1.search import router as search_router
from app.core.security import get_current_user

api_router = APIRouter(prefix="/api/v1")

# Public — no auth required
api_router.include_router(auth_router)

# Protected — every endpoint requires a valid JWT
_auth = [Depends(get_current_user)]
api_router.include_router(search_router, dependencies=_auth)
api_router.include_router(candidates_router, dependencies=_auth)
api_router.include_router(knowledge_base_router, dependencies=_auth)
