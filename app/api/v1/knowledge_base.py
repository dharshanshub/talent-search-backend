from __future__ import annotations

from collections import Counter

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import Response

from app.core.exceptions import BadRequestError, NotFoundError
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
