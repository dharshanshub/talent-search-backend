from __future__ import annotations

import asyncio
from typing import Any

import structlog

from app.core.exceptions import UpstreamServiceError

logger = structlog.get_logger(__name__)

# Pinecone recommends a maximum of 100 vectors per upsert call
UPSERT_BATCH_SIZE = 100


class PineconeStore:
    """Wraps a Pinecone Index; all blocking SDK calls run in a thread pool."""

    def __init__(self, index: Any) -> None:
        self._index = index

    async def upsert(self, vectors: list[dict[str, Any]]) -> None:
        """Upsert vectors in batches of UPSERT_BATCH_SIZE.

        Raises:
            UpstreamServiceError: if Pinecone returns an error on any batch.
        """
        if not vectors:
            logger.debug("upsert_skipped", reason="empty_vectors")
            return
        if self._index is None:
            logger.warning(
                "pinecone_index_none",
                action="upsert_skipped",
                hint="Pinecone failed to connect at startup",
            )
            raise UpstreamServiceError("pinecone", "Pinecone index is not initialised")

        total = len(vectors)
        upserted = 0
        for i in range(0, total, UPSERT_BATCH_SIZE):
            batch = vectors[i : i + UPSERT_BATCH_SIZE]
            try:
                await asyncio.to_thread(self._index.upsert, vectors=batch)
                upserted += len(batch)
                logger.debug(
                    "upserted_batch",
                    batch_size=len(batch),
                    offset=i,
                    total=total,
                )
            except Exception as exc:
                logger.error(
                    "pinecone_upsert_failed",
                    offset=i,
                    batch_size=len(batch),
                    error=str(exc),
                )
                raise UpstreamServiceError("pinecone", f"Upsert batch failed at offset {i}: {exc}") from exc

        logger.info("upsert_complete", total_upserted=upserted)

    async def query(
        self,
        vector: list[float],
        top_k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query the index for nearest neighbours.

        Returns an empty list (graceful degradation) when Pinecone is unavailable.
        Raises UpstreamServiceError on unexpected API errors.
        """
        if self._index is None:
            logger.warning(
                "pinecone_index_none",
                action="query_skipped",
                hint="Pinecone failed to connect at startup — check PINECONE_API_KEY and PINECONE_INDEX_NAME",
            )
            return []

        logger.debug(
            "pinecone_query",
            vector_dim=len(vector),
            top_k=top_k,
            filter=filter,
        )

        try:
            result = await asyncio.to_thread(
                self._index.query,
                vector=vector,
                top_k=top_k,
                filter=filter,
                include_metadata=True,
            )
        except Exception as exc:
            logger.error("pinecone_query_failed", error=str(exc), top_k=top_k)
            raise UpstreamServiceError("pinecone", f"Query failed: {exc}") from exc

        matches = [
            {"id": m.id, "score": m.score, "metadata": m.metadata}
            for m in result.matches
        ]
        logger.info(
            "pinecone_query_done",
            raw_matches=len(matches),
            top_score=round(matches[0]["score"], 4) if matches else None,
        )
        return matches

    async def describe_stats(self) -> dict[str, Any]:
        """Return index stats. Returns empty dict if Pinecone is unavailable."""
        if self._index is None:
            return {}
        try:
            stats = await asyncio.to_thread(self._index.describe_index_stats)
            return stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
        except Exception as exc:
            logger.warning("pinecone_describe_stats_failed", error=str(exc))
            return {}

    async def ping(self) -> bool:
        """Health check — raises RuntimeError if Pinecone is not connected."""
        if self._index is None:
            raise RuntimeError("Pinecone index not initialised")
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._index.describe_index_stats),
                timeout=10.0,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Pinecone ping timed out after 10 s") from exc
        except Exception as exc:
            raise RuntimeError(f"Pinecone ping failed: {exc}") from exc
        return True
