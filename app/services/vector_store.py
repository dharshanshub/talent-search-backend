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

    async def list_candidates(self) -> list[dict[str, Any]]:
        """Return one metadata record per unique candidate.

        Two tiers, merged transparently:

        Tier 1 (fast): Candidates indexed after the profile-vector change have a
          dedicated "profile_{candidate_id}" vector.  list(prefix="profile_") +
          fetch() returns their metadata with minimal URL overhead.

        Tier 2 (legacy): Older candidates (e.g. seed data) only have chunk vectors.
          We use query(filter={"chunk_index": 0}) to find one chunk per candidate —
          the same API path that search uses, so it is always available.  Metadata
          is returned inline; no extra fetch() call is needed.
        """
        if self._index is None:
            return []

        _FETCH_BATCH = 20  # ~20 × 35-char IDs ≈ 700 chars — well under URL limit
        _EMBED_DIM = 1536  # OpenAI text-embedding-3-small

        # ── Tier 1: profile_ vectors ─────────────────────────────────────────
        def _to_str(item: Any) -> str:
            return item if isinstance(item, str) else str(getattr(item, "id", item))

        def _list_profile_ids() -> list[str]:
            ids: list[str] = []
            try:
                for batch in self._index.list(prefix="profile_"):
                    if isinstance(batch, list):
                        ids.extend(_to_str(x) for x in batch)
                    else:
                        ids.append(_to_str(batch))
            except Exception as exc:
                logger.warning("pinecone_list_profile_failed", error=str(exc))
            return ids

        profile_ids = await asyncio.to_thread(_list_profile_ids)
        profiled_cids: set[str] = {vid[len("profile_"):] for vid in profile_ids}

        records: list[dict[str, Any]] = []

        for i in range(0, len(profile_ids), _FETCH_BATCH):
            batch = profile_ids[i : i + _FETCH_BATCH]
            try:
                result = await asyncio.to_thread(self._index.fetch, ids=batch)
                vectors = getattr(result, "vectors", {}) or {}
                for vid in batch:
                    cid = vid[len("profile_"):]
                    meta: dict[str, Any] = dict(getattr(vectors.get(vid), "metadata", {}) or {}) if vid in vectors else {}
                    meta.setdefault("candidate_id", cid)
                    records.append(meta)
            except Exception as exc:
                logger.error("pinecone_fetch_profile_failed", offset=i, error=str(exc))
                raise UpstreamServiceError("pinecone", f"fetch() failed at offset {i}: {exc}") from exc

        # ── Tier 2: legacy candidates via query() ─────────────────────────────
        # list(prefix="") is unreliable in Pinecone SDK v7 on serverless indexes.
        # query() with a chunk_index=0 filter is the same path search uses and is
        # always reliable.  Metadata is returned inline, so no fetch() is needed.
        legacy_added = 0
        try:
            uniform = 1.0 / (_EMBED_DIM ** 0.5)
            dummy_vector = [uniform] * _EMBED_DIM
            result = await asyncio.to_thread(
                self._index.query,
                vector=dummy_vector,
                top_k=10000,
                filter={"chunk_index": {"$eq": 0}},
                include_metadata=True,
            )
            for match in (result.matches or []):
                meta = dict(match.metadata or {})
                cid = meta.get("candidate_id", "")
                if cid and cid not in profiled_cids:
                    meta.setdefault("candidate_id", cid)
                    records.append(meta)
                    profiled_cids.add(cid)
                    legacy_added += 1
        except Exception as exc:
            logger.warning("pinecone_legacy_query_failed", error=str(exc))

        logger.info(
            "list_candidates_done",
            tier1_profile=len(profile_ids),
            tier2_legacy=legacy_added,
            total=len(records),
        )
        return records

    async def list_candidates_page(
        self,
        cursor: str | None,
        limit: int,
        search: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Return one page of candidate metadata.

        Primary path: list_paginated(prefix="profile_") + fetch().
        Fallback path: query(chunk_index=0) + integer offset cursor.

        Pinecone SDK v7 on serverless indexes does not honour prefix filters on
        list/list_paginated, so both return empty even when profile_ vectors exist.
        The fallback uses query() which is always reliable on serverless — the same
        path search and stats already use.

        Args:
            cursor: Pinecone pagination token (primary) or integer offset string
                    (fallback).  None = first page.
            limit:  Records per page (1–100).

        Returns:
            (records, next_cursor) — next_cursor is None on the last page.
        """
        if self._index is None:
            return [], None

        _FETCH_BATCH = 20  # keep URL length < 2 KB
        _EMBED_DIM = 1536

        def _to_str(item: Any) -> str:
            return item if isinstance(item, str) else str(getattr(item, "id", item))

        # ── Search mode: query all, filter by name/title/role, return all matches ──
        # When a search term is provided we skip pagination entirely and return every
        # candidate whose name, title, or role contains the term (case-insensitive).
        # Uses the same query(chunk_index=0) path as stats — always reliable on serverless.
        if search:
            search_lower = search.strip().lower()
            try:
                uniform = 1.0 / (_EMBED_DIM ** 0.5)
                dummy = [uniform] * _EMBED_DIM
                query_result = await asyncio.to_thread(
                    self._index.query,
                    vector=dummy,
                    top_k=10000,
                    filter={"chunk_index": {"$eq": 0}},
                    include_metadata=True,
                )
                all_records = [dict(m.metadata or {}) for m in (query_result.matches or [])]
            except Exception as exc:
                logger.error("kb_search_query_failed", search=search, error=str(exc))
                raise UpstreamServiceError("pinecone", f"Search query failed: {exc}") from exc

            matches = [
                r for r in all_records
                if search_lower in (r.get("name") or "").lower()
                or search_lower in (r.get("title") or "").lower()
                or search_lower in (r.get("role") or "").lower()
            ]
            logger.info("list_candidates_search_done", term=search, matched=len(matches), total_scanned=len(all_records))
            return matches, None

        # ── Primary: list_paginated + fetch ───────────────────────────────────
        def _get_page_primary() -> tuple[list[str], str | None]:
            try:
                result = self._index.list_paginated(
                    prefix="profile_",
                    limit=limit,
                    pagination_token=cursor if cursor and not cursor.isdigit() else None,
                )
                ids = [_to_str(v) for v in (result.vectors or [])]
                next_cur = result.pagination.next if result.pagination else None
                return ids, next_cur
            except AttributeError:
                logger.warning("list_paginated_unavailable", hint="upgrade pinecone>=3.0")
                return [], None

        page_ids, list_next_cursor = await asyncio.to_thread(_get_page_primary)

        if page_ids:
            records: list[dict[str, Any]] = []
            for i in range(0, len(page_ids), _FETCH_BATCH):
                batch = page_ids[i : i + _FETCH_BATCH]
                try:
                    result = await asyncio.to_thread(self._index.fetch, ids=batch)
                    vectors = getattr(result, "vectors", {}) or {}
                    for vid in batch:
                        cid = vid[len("profile_"):]
                        meta: dict[str, Any] = (
                            dict(getattr(vectors.get(vid), "metadata", {}) or {})
                            if vid in vectors else {}
                        )
                        meta.setdefault("candidate_id", cid)
                        records.append(meta)
                except Exception as exc:
                    logger.error("kb_page_fetch_failed", batch_size=len(batch), error=str(exc))
                    raise UpstreamServiceError("pinecone", f"fetch() failed for page: {exc}") from exc

            logger.info("list_candidates_page_done", returned=len(records), has_next=list_next_cursor is not None, method="list_paginated")
            return records, list_next_cursor

        # ── Fallback: query(chunk_index=0) + integer offset cursor ────────────
        # list_paginated returned empty — Pinecone serverless v7 ignores prefix
        # filters on list operations.  query() is always reliable: same path as
        # search and stats.  We use a fixed uniform dummy vector so results are
        # deterministic across pages.
        logger.info("list_paginated_empty_fallback", hint="pinecone serverless prefix filter unreliable")

        offset = int(cursor) if cursor and cursor.isdigit() else 0

        try:
            uniform = 1.0 / (_EMBED_DIM ** 0.5)
            dummy = [uniform] * _EMBED_DIM
            query_result = await asyncio.to_thread(
                self._index.query,
                vector=dummy,
                top_k=10000,
                filter={"chunk_index": {"$eq": 0}},
                include_metadata=True,
            )
            all_records = [dict(m.metadata or {}) for m in (query_result.matches or [])]
        except Exception as exc:
            logger.error("kb_candidates_query_fallback_failed", error=str(exc))
            raise UpstreamServiceError("pinecone", f"Candidates query failed: {exc}") from exc

        page = all_records[offset : offset + limit]
        next_cur = str(offset + limit) if offset + limit < len(all_records) else None

        logger.info("list_candidates_page_done", returned=len(page), has_next=next_cur is not None, method="query_offset", total_found=len(all_records))
        return page, next_cur

    async def get_stats_sample(
        self,
        max_samples: int = 10000,
    ) -> tuple[list[dict[str, Any]], int]:
        """Collect metadata for stats aggregation and count total profile vectors.

        Two parallel Pinecone operations:
          1. query(filter={chunk_index:0}, top_k=max_samples) — returns metadata
             for seniority, skills, experience, and last_added_at aggregation.
             At >max_samples candidates the stats are computed from a representative
             sample; the sample size is surfaced in the response so the UI can note it.
          2. list(prefix="profile_") — counts IDs only (no metadata, minimal bandwidth).
             Accurate total regardless of sample size.

        Returns:
            (metadata_sample, total_profile_count)
        """
        if self._index is None:
            return [], 0

        _EMBED_DIM = 1536

        # Run both operations concurrently
        async def _query_metadata() -> list[dict[str, Any]]:
            try:
                uniform = 1.0 / (_EMBED_DIM ** 0.5)
                dummy = [uniform] * _EMBED_DIM
                result = await asyncio.to_thread(
                    self._index.query,
                    vector=dummy,
                    top_k=max_samples,
                    filter={"chunk_index": {"$eq": 0}},
                    include_metadata=True,
                )
                return [dict(m.metadata or {}) for m in (result.matches or [])]
            except Exception as exc:
                logger.warning("stats_metadata_query_failed", error=str(exc))
                return []

        def _count_profile_ids() -> int:
            count = 0
            try:
                for batch in self._index.list(prefix="profile_"):
                    count += len(batch) if isinstance(batch, list) else 1
            except Exception as exc:
                logger.warning("stats_count_profiles_failed", error=str(exc))
            return count

        metadata_task = asyncio.create_task(_query_metadata())
        total = await asyncio.to_thread(_count_profile_ids)
        metadata_sample = await metadata_task

        # If profile_ count failed, fall back to number of query matches
        if total == 0:
            total = len(metadata_sample)

        logger.info(
            "stats_sample_done",
            sampled=len(metadata_sample),
            total_profiles=total,
        )
        return metadata_sample, total

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
