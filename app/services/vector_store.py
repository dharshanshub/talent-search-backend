from __future__ import annotations

import asyncio
import re
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

    async def list_candidates(self) -> list[dict[str, Any]]:
        """Return one metadata record per unique candidate by listing all vector IDs,
        grouping by candidate_id, then fetching chunk_0 for each.

        Returns an empty list if Pinecone is unavailable or the index type does not
        support list() (pod-based indexes). Each record includes all profile metadata
        plus a `chunks` count.

        Raises UpstreamServiceError on unexpected fetch failures.
        """
        if self._index is None:
            return []

        # ── Step 1: collect all vector IDs ───────────────────────────────────
        # Pinecone v7 list() yields ListItem objects with an .id attribute.
        # Older SDK versions yielded plain strings or lists of strings.
        def _extract_id(item) -> str:
            if isinstance(item, str):
                return item
            return str(getattr(item, "id", item))

        def _collect_ids() -> list[str]:
            ids: list[str] = []
            for batch in self._index.list():
                if isinstance(batch, list):
                    ids.extend(_extract_id(item) for item in batch)
                elif isinstance(batch, str):
                    ids.append(batch)
                else:
                    # Single ListItem (Pinecone v7 yields items directly)
                    ids.append(_extract_id(batch))
            return ids

        try:
            all_ids: list[str] = await asyncio.to_thread(_collect_ids)
        except Exception as exc:
            logger.error("pinecone_list_failed", error=str(exc))
            raise UpstreamServiceError("pinecone", f"list() failed: {exc}") from exc

        if not all_ids:
            return []

        # ── Step 2: extract unique candidate_ids and count chunks ─────────────
        chunk_counts: dict[str, int] = {}
        for vid in all_ids:
            cid = re.sub(r"_chunk_\d+$", "", vid)
            chunk_counts[cid] = chunk_counts.get(cid, 0) + 1

        # ── Step 3: fetch metadata from chunk_0 for each candidate ───────────
        candidate_ids = list(chunk_counts.keys())
        chunk0_ids = [f"{cid}_chunk_0" for cid in candidate_ids]

        FETCH_BATCH = 100
        records: list[dict[str, Any]] = []

        for i in range(0, len(chunk0_ids), FETCH_BATCH):
            batch_ids = chunk0_ids[i : i + FETCH_BATCH]
            batch_cids = candidate_ids[i : i + FETCH_BATCH]
            try:
                result = await asyncio.to_thread(
                    self._index.fetch, ids=batch_ids
                )
            except Exception as exc:
                logger.error("pinecone_fetch_failed", offset=i, error=str(exc))
                raise UpstreamServiceError("pinecone", f"fetch() failed at offset {i}: {exc}") from exc

            vectors = getattr(result, "vectors", {}) or {}
            for cid, chunk0_id in zip(batch_cids, batch_ids):
                meta = {}
                if chunk0_id in vectors:
                    vec = vectors[chunk0_id]
                    meta = getattr(vec, "metadata", {}) or {}
                records.append({
                    "candidate_id": cid,
                    "chunks": chunk_counts[cid],
                    **meta,
                })

        logger.info("list_candidates_done", total=len(records))
        return records

    async def delete_by_candidate_id(self, candidate_id: str) -> None:
        """Delete all vectors for a candidate using a metadata filter.

        This removes every chunk stored for the given candidate_id.

        Raises:
            UpstreamServiceError: if Pinecone is not initialised or the delete fails.
        """
        if self._index is None:
            raise UpstreamServiceError("pinecone", "Pinecone index is not initialised")

        try:
            await asyncio.to_thread(
                self._index.delete,
                filter={"candidate_id": {"$eq": candidate_id}},
            )
            logger.info("pinecone_deleted_candidate", candidate_id=candidate_id)
        except Exception as exc:
            logger.error("pinecone_delete_failed", candidate_id=candidate_id, error=str(exc))
            raise UpstreamServiceError("pinecone", f"Delete failed for '{candidate_id}': {exc}") from exc

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
