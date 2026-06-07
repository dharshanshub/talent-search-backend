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
        """Return one metadata record per unique candidate.

        Architecture — two tiers handled transparently:

        Tier 1 (new): candidates indexed after the profile-vector change have a
          dedicated vector with id = "profile_{candidate_id}".  list() with
          prefix="profile_" returns exactly one ID per candidate — tiny URL, fast.

        Tier 2 (legacy): older candidates only have chunk vectors.  We detect
          them by finding _chunk_0 IDs whose candidate_id has no profile_ vector,
          then fetch those instead.

        fetch() is batched at 20 IDs max to stay well under Pinecone's URL limit.
        """
        if self._index is None:
            return []

        _FETCH_BATCH = 20  # safe ceiling: ~20 × 35-char IDs ≈ 700 chars in query string

        # ── helper: Pinecone v7 yields ListItem objects; older SDKs yield strings ──
        def _to_str(item: Any) -> str:
            return item if isinstance(item, str) else str(getattr(item, "id", item))

        def _list_prefix(prefix: str) -> list[str]:
            ids: list[str] = []
            for batch in self._index.list(prefix=prefix):
                if isinstance(batch, list):
                    ids.extend(_to_str(x) for x in batch)
                else:
                    ids.append(_to_str(batch))
            return ids

        # ── Step 1: collect profile_ IDs (Tier 1) ────────────────────────────
        try:
            profile_ids: list[str] = await asyncio.to_thread(_list_prefix, "profile_")
        except Exception as exc:
            logger.error("pinecone_list_profile_failed", error=str(exc))
            raise UpstreamServiceError("pinecone", f"list(prefix='profile_') failed: {exc}") from exc

        profiled_cids: set[str] = {vid[len("profile_"):] for vid in profile_ids}

        # ── Step 2: find legacy candidates (Tier 2) ──────────────────────────
        # Only scan if there might be legacy data (heuristic: always check once)
        legacy_fetch_ids: list[str] = []
        try:
            all_ids: list[str] = await asyncio.to_thread(_list_prefix, "")
        except Exception as exc:
            logger.warning("pinecone_list_all_failed", error=str(exc))
            all_ids = []

        for vid in all_ids:
            if re.search(r"_chunk_0$", vid):
                cid = re.sub(r"_chunk_0$", "", vid)
                if cid not in profiled_cids:
                    legacy_fetch_ids.append(vid)  # chunk_0 id, used as fetch key

        logger.info(
            "list_candidates_tiers",
            profile_count=len(profile_ids),
            legacy_count=len(legacy_fetch_ids),
        )

        # ── Step 3: batch-fetch metadata for both tiers ───────────────────────
        def _extract_meta(fetch_result: Any, requested_ids: list[str]) -> list[dict[str, Any]]:
            vectors = getattr(fetch_result, "vectors", {}) or {}
            rows = []
            for vid in requested_ids:
                meta: dict[str, Any] = {}
                if vid in vectors:
                    meta = dict(getattr(vectors[vid], "metadata", {}) or {})
                # Derive candidate_id from the vector id
                if vid.startswith("profile_"):
                    cid = vid[len("profile_"):]
                else:
                    cid = re.sub(r"_chunk_0$", "", vid)
                meta.setdefault("candidate_id", cid)
                rows.append(meta)
            return rows

        records: list[dict[str, Any]] = []

        for fetch_list in (profile_ids, legacy_fetch_ids):
            for i in range(0, len(fetch_list), _FETCH_BATCH):
                batch = fetch_list[i : i + _FETCH_BATCH]
                try:
                    result = await asyncio.to_thread(self._index.fetch, ids=batch)
                except Exception as exc:
                    logger.error("pinecone_fetch_failed", offset=i, batch_size=len(batch), error=str(exc))
                    raise UpstreamServiceError("pinecone", f"fetch() failed at offset {i}: {exc}") from exc
                records.extend(_extract_meta(result, batch))

        logger.info("list_candidates_done", total=len(records))
        return records

    async def upsert_profile_vector(
        self,
        candidate_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Upsert a single profile_ vector for a candidate (used by backfill)."""
        vector = {
            "id":       f"profile_{candidate_id}",
            "values":   embedding,
            "metadata": {**metadata, "chunk_type": "profile", "candidate_id": candidate_id},
        }
        await self.upsert([vector])

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
