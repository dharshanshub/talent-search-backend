from __future__ import annotations

from collections import Counter

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import BaseModel

from app.core.exceptions import BadRequestError
from app.core.logging import get_correlation_id
from app.schemas.knowledge_base import (
    CandidateRecord,
    KnowledgeBaseResponse,
    KnowledgeBaseStats,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/knowledge-base", tags=["knowledge-base"])

_SENIORITY_ORDER = ["Junior", "Mid-Level", "Senior", "Staff", "Principal"]


def _parse_record(raw: dict) -> CandidateRecord:
    """Coerce raw Pinecone metadata dict into a CandidateRecord."""
    skills_raw = raw.get("skills", "")
    skills = [s.strip() for s in skills_raw.split(",") if s.strip()] if isinstance(skills_raw, str) else list(skills_raw)

    industries_raw = raw.get("industries", "")
    industries = [i.strip() for i in industries_raw.split(",") if i.strip()] if isinstance(industries_raw, str) else list(industries_raw)

    # Prefer indexed_at (full ISO datetime) — fall back to last_updated (date string)
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


def _build_stats(candidates: list[CandidateRecord]) -> KnowledgeBaseStats:
    total = len(candidates)

    # Seniority distribution — preserve defined order, omit zeros
    seniority_counts = Counter(c.seniority for c in candidates if c.seniority)
    seniority_distribution = {
        s: seniority_counts[s]
        for s in _SENIORITY_ORDER
        if seniority_counts[s] > 0
    }

    # Average experience
    experiences = [c.years_experience for c in candidates]
    avg_exp = round(sum(experiences) / len(experiences), 1) if experiences else 0.0

    # Last added — max of indexed_at strings (ISO sorts lexicographically)
    dated = [c.indexed_at for c in candidates if c.indexed_at]
    last_added_at = max(dated) if dated else None

    # Top 10 skills across the whole pool
    all_skills: list[str] = []
    for c in candidates:
        all_skills.extend(c.skills)
    top_skills = [skill for skill, _ in Counter(all_skills).most_common(10)]

    return KnowledgeBaseStats(
        total_profiles=total,
        seniority_distribution=seniority_distribution,
        avg_experience_years=avg_exp,
        last_added_at=last_added_at,
        top_skills=top_skills,
    )


@router.get("", response_model=KnowledgeBaseResponse)
async def list_knowledge_base(request: Request) -> KnowledgeBaseResponse:
    """List all candidates in the knowledge base with pool-level stats.

    Reads metadata from Pinecone — no separate database required.
    Older candidates (seeded before indexed_at was added) fall back to last_updated.
    """
    request_id = get_correlation_id()
    logger.info("knowledge_base_list_request", request_id=request_id)

    vector_store = request.app.state.vector_store
    raw_records = await vector_store.list_candidates()

    candidates = [_parse_record(r) for r in raw_records]
    stats = _build_stats(candidates)

    logger.info(
        "knowledge_base_list_done",
        total=stats.total_profiles,
        request_id=request_id,
    )
    return KnowledgeBaseResponse(stats=stats, candidates=candidates)


@router.delete("/{candidate_id}", status_code=204)
async def delete_candidate(candidate_id: str, request: Request) -> Response:
    """Remove a candidate from Pinecone and Azure Blob Storage.

    Deletes all indexed chunks for the candidate_id from Pinecone, then
    deletes the source PDF from Blob Storage. Safe to call even if the blob
    no longer exists.

    Raises:
        BadRequestError: if the candidate_id contains unsafe characters.
        NotFoundError: if the candidate has no vectors in Pinecone.
    """
    request_id = get_correlation_id()

    if not all(c.isalnum() or c in "_-" for c in candidate_id):
        raise BadRequestError("Invalid candidate ID")

    logger.info("knowledge_base_delete_request", candidate_id=candidate_id, request_id=request_id)

    vector_store = request.app.state.vector_store
    blob_service = request.app.state.blob_service

    # Delete from Pinecone (all chunks for this candidate)
    await vector_store.delete_by_candidate_id(candidate_id)

    # Delete source PDF from Blob Storage (no-op if not configured or already gone)
    blob_name = f"{candidate_id}.pdf"
    await blob_service.delete(blob_name)

    logger.info(
        "knowledge_base_delete_done",
        candidate_id=candidate_id,
        request_id=request_id,
    )
    return Response(status_code=204)


class BackfillResponse(BaseModel):
    created: int
    skipped: int
    failed: int


@router.post("/backfill", response_model=BackfillResponse)
async def backfill_profile_vectors(request: Request) -> BackfillResponse:
    """One-time operation: create profile_ vectors for legacy candidates.

    Legacy candidates (e.g. seeded data) have chunk vectors but no profile_ vector,
    so they won't appear in the dashboard's Tier-1 fast path.

    Strategy:
      1. list(prefix="profile_") — collect already-profiled candidate IDs.
      2. query(filter={"chunk_index": 0}, include_values=True) — find one chunk per
         legacy candidate, with the embedding already attached in the response.
         This avoids list(prefix="") which is unreliable on Pinecone SDK v7 serverless.
      3. Upsert a profile_{candidate_id} vector for each unprocessed candidate.

    Safe to call multiple times — already-profiled candidates are skipped.
    """
    import asyncio as _asyncio

    request_id = get_correlation_id()
    logger.info("backfill_start", request_id=request_id)

    vector_store = request.app.state.vector_store
    index = vector_store._index

    if index is None:
        raise BadRequestError("Pinecone is not connected")

    _EMBED_DIM = 1536   # OpenAI text-embedding-3-small
    _UPSERT_BATCH = 100

    # ── Step 1: which candidates already have profile vectors? ────────────────
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

    profile_ids = await _asyncio.to_thread(_list_profile_ids_sync)
    profiled_cids: set[str] = {vid[len("profile_"):] for vid in profile_ids}

    # ── Step 2: find legacy chunk_0 vectors via query() ───────────────────────
    # include_values=True so we can reuse the embedding for the new profile_ vector.
    try:
        uniform = 1.0 / (_EMBED_DIM ** 0.5)
        dummy = [uniform] * _EMBED_DIM
        query_result = await _asyncio.to_thread(
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

    # ── Step 3: build list of new profile vectors to upsert ───────────────────
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

    # ── Step 4: upsert in batches ─────────────────────────────────────────────
    created = failed = 0
    skipped = already_profiled

    for i in range(0, len(to_create), _UPSERT_BATCH):
        batch = to_create[i : i + _UPSERT_BATCH]
        try:
            await _asyncio.to_thread(index.upsert, vectors=batch)
            created += len(batch)
            logger.info("backfill_batch_upserted", count=len(batch), offset=i)
        except Exception as exc:
            logger.error("backfill_upsert_failed", offset=i, error=str(exc))
            failed += len(batch)

    logger.info(
        "backfill_done",
        created=created, skipped=skipped, failed=failed,
        request_id=request_id,
    )
    return BackfillResponse(created=created, skipped=skipped, failed=failed)
