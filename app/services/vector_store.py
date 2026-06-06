from __future__ import annotations

import asyncio
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Pinecone recommends max 100 vectors per upsert batch
UPSERT_BATCH_SIZE = 100


class PineconeStore:
    """Wraps a Pinecone Index; all blocking SDK calls run in a thread pool."""

    def __init__(self, index: Any) -> None:
        self._index = index

    async def upsert(self, vectors: list[dict[str, Any]]) -> None:
        if not vectors or self._index is None:
            return
        # Batch into chunks of UPSERT_BATCH_SIZE
        for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
            batch = vectors[i : i + UPSERT_BATCH_SIZE]
            await asyncio.to_thread(self._index.upsert, vectors=batch)
            logger.debug("upserted_batch", batch_size=len(batch), offset=i)

    async def query(
        self,
        vector: list[float],
        top_k: int,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self._index is None:
            logger.warning("pinecone_index_none", action="query_skipped",
                           hint="Pinecone failed to connect at startup — check PINECONE_API_KEY and PINECONE_INDEX_NAME")
            return []

        logger.debug("pinecone_query",
                     vector_dim=len(vector),
                     top_k=top_k,
                     filter=filter)

        result = await asyncio.to_thread(
            self._index.query,
            vector=vector,
            top_k=top_k,
            filter=filter,
            include_metadata=True,
        )
        matches = [
            {"id": m.id, "score": m.score, "metadata": m.metadata}
            for m in result.matches
        ]
        logger.info("pinecone_query_done",
                    raw_matches=len(matches),
                    top_score=round(matches[0]["score"], 4) if matches else None)
        return matches

    async def describe_stats(self) -> dict[str, Any]:
        """Returns index stats dict (total_vector_count, namespaces, etc.)."""
        if self._index is None:
            return {}
        stats = await asyncio.to_thread(self._index.describe_index_stats)
        return stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)

    async def ping(self) -> bool:
        if self._index is None:
            raise RuntimeError("Pinecone index not initialised")
        await asyncio.to_thread(self._index.describe_index_stats)
        return True
