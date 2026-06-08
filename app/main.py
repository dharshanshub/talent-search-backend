from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

from app.api.health import router as health_router
from app.api.router import api_router
from app.core.config import get_settings
from app.core.error_handlers import (
    app_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.core.exceptions import AppException
from app.core.logging import configure_logging
from app.middleware.correlation import CorrelationMiddleware
from app.services.agent import AgentService
from app.services.blob_storage import BlobStorageService
from app.services.embeddings import OpenAIEmbedder
from app.services.llm import OpenAILLM
from app.services.query_understanding import QueryUnderstandingService
from app.services.screening import ScreeningService
from app.services.search_service import SearchService
from app.services.vector_store import PineconeStore

logger = structlog.get_logger(__name__)


def _init_pinecone(settings) -> object | None:
    """Initialise Pinecone index synchronously. Returns index or None on failure."""
    try:
        from pinecone import Pinecone  # type: ignore[import]
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)
        logger.info("pinecone_connected", index=settings.pinecone_index_name)
        return index
    except Exception as exc:
        logger.warning("pinecone_unavailable", reason=str(exc))
        return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    logger.info("startup_begin", app_env=settings.app_env)

    # OpenAI async client
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    app.state.openai_client = openai_client

    # Pinecone — run blocking init in thread so it doesn't block the event loop
    try:
        index = await asyncio.wait_for(
            asyncio.to_thread(_init_pinecone, settings),
            timeout=35.0,
        )
    except asyncio.TimeoutError:
        logger.warning("pinecone_timeout", seconds=35)
        index = None

    vector_store = PineconeStore(index=index)
    app.state.vector_store = vector_store

    # Log index stats so we know immediately if the index is empty
    if index is not None:
        try:
            stats = await asyncio.wait_for(
                asyncio.to_thread(index.describe_index_stats), timeout=8.0
            )
            total_vectors = getattr(stats, "total_vector_count", "unknown")
            logger.info("pinecone_index_stats", total_vectors=total_vectors,
                        index=settings.pinecone_index_name)
            if total_vectors == 0:
                logger.warning("pinecone_index_empty",
                               hint="Upload resumes via the Screen Resume feature")
        except Exception as exc:
            logger.warning("pinecone_stats_failed", reason=str(exc))

    # Service layer
    embedder = OpenAIEmbedder(
        client=openai_client,
        model=settings.openai_embedding_model,
        dimensions=settings.embedding_dim,
    )
    llm = OpenAILLM(client=openai_client, model=settings.openai_llm_model)
    query_understanding = QueryUnderstandingService(llm_client=llm)
    app.state.embedder = embedder  # exposed for backfill endpoint
    search_service = SearchService(
        embedder=embedder,
        vector_store=vector_store,
        query_understanding=query_understanding,
        llm=llm,
        top_k=settings.top_k,
    )
    app.state.search_service = search_service

    # Agentic layer — sits in front of search, handles routing + streaming
    agent = AgentService(
        openai_client=openai_client,
        search_service=search_service,
        model=settings.openai_llm_model,
    )
    app.state.agent = agent

    screening_service = ScreeningService(
        llm=llm,
        embedder=embedder,
        vector_store=vector_store,
    )
    app.state.screening_service = screening_service

    # Azure Blob Storage — optional; empty connection string = local disk fallback
    blob_service = BlobStorageService(
        connection_string=settings.azure_storage_connection_string,
        container=settings.azure_storage_container,
    )
    app.state.blob_service = blob_service

    logger.info(
        "startup_complete",
        pinecone_ready=index is not None,
        blob_storage_ready=blob_service.available,
    )
    yield

    logger.info("shutdown_begin")
    await openai_client.close()
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(app_env=settings.app_env, log_level=settings.log_level)

    app = FastAPI(
        title="Talent Search RAG",
        version="1.0.0",
        docs_url="/docs" if settings.app_env == "dev" else None,
        redoc_url="/redoc" if settings.app_env == "dev" else None,
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(CorrelationMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # Routers
    app.include_router(health_router)
    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "Talent Search RAG API",
            "version": "1.0.0",
            "status": "ok",
            "docs": "/docs" if settings.app_env == "dev" else None,
            "health": "/health",
            "api": "/api/v1",
            "note": "This is the backend API. Open the frontend app URL to use the UI.",
        }

    return app


app = create_app()
