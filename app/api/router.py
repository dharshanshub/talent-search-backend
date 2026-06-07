from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.candidates import router as candidates_router
from app.api.v1.knowledge_base import router as knowledge_base_router
from app.api.v1.search import router as search_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(search_router)
api_router.include_router(candidates_router)
api_router.include_router(knowledge_base_router)
