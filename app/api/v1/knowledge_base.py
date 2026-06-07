from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel

from app.core.exceptions import BadRequestError, NotFoundError
from app.core.logging import get_correlation_id
from app.schemas.knowledge_base import (
    CandidateRecord,
    CandidatesPageResponse,
    KnowledgeBaseStats,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/knowledge-base", tags=["knowledge-base"])

_SENIORITY_ORDER = ["Junior", "Mid-Level", "Senior", "Staff", "Principal"]
_STATS_TTL_SECONDS = 300  # 5-minute cache


# ── Shared helpers ────────────────────────────────────────────────────────────

def _parse_record(raw: dict) -> CandidateRecord:
    skills_raw = raw.get("skills", "")
    skills = (
        [s.strip() for s in skills_raw.split(",") if s.strip()]
        if isinstance(skills_raw, str)
        else list(skills_raw)
    )
    industries_raw = raw.get("industries", "")
    industries = (
        [i.strip() for i in industries_raw.split(",") if i.strip()]
        if isinstance(industries_raw, str)
        else list(industries_raw)
    )
    indexed_at = raw.get("indexed_at") or raw.get("last_updated") or ""
    return CandidateRecord(
        candidate_id=raw.get("candidate_id", ""),
        name=raw.get("name", "Unknown"),
        title=raw.get("title", ""),
        role=raw.get("role", ""),
        seniority=raw.get("seniority", ""),
        location=raw.get("location", ""),
        years_experience=int(raw.get("years_experience", 0)),
        skills=skills,
        industries=industries,
        blob_filename=raw.get("blob_filename") or None,
        indexed_at=indexed_at,
    )


def _compute_stats(metadata_sample: list[dict], total: int) -> KnowledgeBaseStats:
    records = [_parse_record(r) for r in metadata_sample]
    sampled = len(records)

    seniority_counts = Counter(c.seniority for c in records if c.seniority)
    seniority_distribution = {
        s: seniority_counts[s]
        for s in _SENIORITY_ORDER
        if seniority_counts[s] > 0
    }

    experiences = [c.years_experience for c in records]
    avg_exp = round(sum(experiences) / len(experiences), 1) if experiences else 0.0

    dated = [c.indexed_at for c in records if c.indexed_at]
    last_added_at = max(dated) if dated else None

    all_skills: list[str] = []
    for c in records:
        all_skills.extend(c.skills)
    top_skills = [skill for skill, _ in Counter(all_skills).most_common(10)]

    return KnowledgeBaseStats(
        total_profiles=total,
        seniority_distribution=seniority_distribution,
        avg_experience_years=avg_exp,
        last_added_at=last_added_at,
        top_skills=top_skills,
        is_sampled=sampled < total,
    )


# ── Stats cache helpers ───────────────────────────────────────────────────────

def _stats_cache(request: Request) -> dict:
    """Lazy-init the stats cache on app.state."""
    if not hasattr(request.app.state, "kb_stats_cache"):
        request.app.state.kb_stats_cache = {
            "stats": None,
            "total": -1,
            "cached_at": None,
        }
    return request.app.state.kb_stats_cache


def _cache_is_fresh(cache: dict) -> bool:
    if cache["cached_at"] is None or cache["stats"] is None:
        return False
    age = (datetime.now(timezone.utc) - cache["cached_at"]).total_seconds()
    return age < _STATS_TTL_SECONDS


async def _refresh_stats(request: Request) -> KnowledgeBaseStats:
    """Full stats computation — runs on cache miss, result stored in app.state."""
    vector_store = request.app.state.vector_store
    metadata_sample, total = await vector_store.get_stats_sample()
    stats = _compute_stats(metadata_sample, total)

    cache = _stats_cache(request)
    cache["stats"] = stats
    cache["total"] = total
    cache["cached_at"] = datetime.now(timezone.utc)

    logger.info("kb_stats_cache_refreshed", total=total, sampled=len(metadata_sample))
    return stats


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=KnowledgeBaseStats)
async def get_stats(request: Request) -> KnowledgeBaseStats:
    """Pool-level stats for the knowledge base dashboard.

    Cached in memory for 5 minutes.  The first request after startup (or after
    the TTL expires) triggers a full computation — subsequent calls are instant.

    Stats are derived from a query sample of up to 10 000 candidates; at larger
    pool sizes seniority/skills/experience are approximate and is_sampled=True.
    total_profiles is always exact (counted from profile_ vector IDs).
    """
    cache = _stats_cache(request)

    if _cache_is_fresh(cache):
        logger.debug("kb_stats_cache_hit")
        return cache["stats"]

    logger.info("kb_stats_cache_miss", reason="stale_or_empty")
    return await _refresh_stats(request)


@router.get("/candidates", response_model=CandidatesPageResponse)
async def list_candidates_page(
    request: Request,
    cursor: str | None = Query(default=None, description="Pagination token from previous response"),
    limit: int = Query(default=20, ge=1, le=100, description="Records per page"),
) -> CandidatesPageResponse:
    """Cursor-paginated candidate list — O(1) per page regardless of pool size.

    Pass the returned next_cursor as cursor on the next request to advance.
    next_cursor=null means you are on the last page.

    total comes from the stats cache (instant if /stats was called first).
    If stats have not been computed yet it returns -1 — the UI should call
    /stats in parallel to get the accurate count.
    """
    request_id = get_correlation_id()
    vector_store = request.app.state.vector_store

    raw_records, next_cursor = await vector_store.list_candidates_page(
        cursor=cursor,
        limit=limit,
    )

    candidates = [_parse_record(r) for r in raw_records]

    # Use cached total if available — avoids an extra full scan per page request
    cache = _stats_cache(request)
    total = cache["total"] if _cache_is_fresh(cache) else -1

    logger.info(
        "kb_candidates_page_done",
        returned=len(candidates),
        has_next=next_cursor is not None,
        total_cached=total,
        request_id=request_id,
    )
    return CandidatesPageResponse(
        candidates=candidates,
        next_cursor=next_cursor,
        total=total,
    )


@router.delete("/{candidate_id}", status_code=204)
async def delete_candidate(candidate_id: str, request: Request) -> Response:
    """Remove a candidate from Pinecone and Azure Blob Storage.

    Also invalidates the stats cache so the next /stats call reflects the deletion.
    """
    if not all(c.isalnum() or c in "_-" for c in candidate_id):
        raise BadRequestError("Invalid candidate ID")

    request_id = get_correlation_id()
    logger.info("knowledge_base_delete_request", candidate_id=candidate_id, request_id=request_id)

    vector_store = request.app.state.vector_store
    blob_service = request.app.state.blob_service

    await vector_store.delete_by_candidate_id(candidate_id)

    blob_name = f"{candidate_id}.pdf"
    await blob_service.delete(blob_name)

    # Invalidate stats cache — total_profiles has changed
    cache = _stats_cache(request)
    cache["cached_at"] = None

    logger.info("knowledge_base_delete_done", candidate_id=candidate_id, request_id=request_id)
    return Response(status_code=204)


# ── Resume URL ────────────────────────────────────────────────────────────────

class ResumeUrlResponse(BaseModel):
    url: str
    expires_in: int


@router.get("/{candidate_id}/resume", response_model=ResumeUrlResponse)
async def get_resume_url(candidate_id: str, request: Request) -> ResumeUrlResponse:
    """Generate a 15-minute SAS URL for viewing a candidate's source PDF inline."""
    if not all(c.isalnum() or c in "_-" for c in candidate_id):
        raise BadRequestError("Invalid candidate ID")

    blob_service = request.app.state.blob_service

    if not blob_service.available:
        raise NotFoundError("Blob Storage is not configured — no PDF available")

    url = await blob_service.get_sas_url(f"{candidate_id}.pdf", expiry_minutes=15)
    if url is None:
        raise NotFoundError("No PDF found for this candidate")

    logger.info("resume_url_generated", candidate_id=candidate_id)
    return ResumeUrlResponse(url=url, expires_in=900)


# ── Backfill ──────────────────────────────────────────────────────────────────

class BackfillResponse(BaseModel):
    created: int
    skipped: int
    failed: int


@router.post("/backfill", response_model=BackfillResponse)
async def backfill_profile_vectors(request: Request) -> BackfillResponse:
    """One-time operation: create profile_ vectors for legacy candidates.

    Legacy candidates (e.g. seeded data) have chunk vectors but no profile_ vector,
    so they won't appear in the paginated candidates list.

    Strategy:
      1. list(prefix="profile_") — collect already-profiled candidate IDs.
      2. query(filter={"chunk_index": 0}, include_values=True) — find one chunk per
         legacy candidate with its embedding attached.
      3. Upsert profile_{candidate_id} vectors in batches of 100.

    Safe to call multiple times — already-profiled candidates are skipped.
    After completion, invalidates the stats cache.
    """
    request_id = get_correlation_id()
    logger.info("backfill_start", request_id=request_id)

    vector_store = request.app.state.vector_store
    index = vector_store._index

    if index is None:
        raise BadRequestError("Pinecone is not connected")

    _EMBED_DIM = 1536
    _UPSERT_BATCH = 100

    def _to_str(item) -> str:
        return item if isinstance(item, str) else str(getattr(item, "id", item))

    def _list_profile_ids_sync() -> list[str]:
        ids: list[str] = []
        try:
            for batch in index.list(prefix="profile_"):
                if isinstance(batch, list):
                    ids.extend(_to_str(x) for x in batch)
                else:
                    ids.append(_to_str(batch))
        except Exception as exc:
            logger.warning("backfill_list_profile_failed", error=str(exc))
        return ids

    profile_ids = await asyncio.to_thread(_list_profile_ids_sync)
    profiled_cids: set[str] = {vid[len("profile_"):] for vid in profile_ids}

    try:
        uniform = 1.0 / (_EMBED_DIM ** 0.5)
        dummy = [uniform] * _EMBED_DIM
        query_result = await asyncio.to_thread(
            index.query,
            vector=dummy,
            top_k=10000,
            filter={"chunk_index": {"$eq": 0}},
            include_metadata=True,
            include_values=True,
        )
    except Exception as exc:
        logger.error("backfill_query_failed", error=str(exc), request_id=request_id)
        raise BadRequestError(f"Pinecone query failed during backfill: {exc}") from exc

    to_create: list[dict] = []
    already_profiled = 0

    for match in (query_result.matches or []):
        meta = dict(match.metadata or {})
        cid = meta.get("candidate_id", "")
        values = getattr(match, "values", None) or []
        if not cid:
            continue
        if cid in profiled_cids:
            already_profiled += 1
            continue
        if not values:
            logger.warning("backfill_no_embedding", candidate_id=cid)
            continue
        to_create.append({
            "id":       f"profile_{cid}",
            "values":   values,
            "metadata": {**meta, "chunk_type": "profile", "candidate_id": cid},
        })

    logger.info(
        "backfill_candidates",
        to_create=len(to_create),
        already_profiled=already_profiled,
        request_id=request_id,
    )

    created = failed = 0
    skipped = already_profiled

    for i in range(0, len(to_create), _UPSERT_BATCH):
        batch = to_create[i : i + _UPSERT_BATCH]
        try:
            await asyncio.to_thread(index.upsert, vectors=batch)
            created += len(batch)
            logger.info("backfill_batch_upserted", count=len(batch), offset=i)
        except Exception as exc:
            logger.error("backfill_upsert_failed", offset=i, error=str(exc))
            failed += len(batch)

    # Invalidate stats cache — profile count has changed
    cache = _stats_cache(request)
    cache["cached_at"] = None

    logger.info(
        "backfill_done",
        created=created, skipped=skipped, failed=failed,
        request_id=request_id,
    )
    return BackfillResponse(created=created, skipped=skipped, failed=failed)
