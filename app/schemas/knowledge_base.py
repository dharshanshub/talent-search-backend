from __future__ import annotations

from pydantic import BaseModel


class CandidateRecord(BaseModel):
    candidate_id: str
    name: str
    title: str
    role: str
    seniority: str
    location: str
    years_experience: int
    skills: list[str]
    industries: list[str]
    blob_filename: str | None
    indexed_at: str


class KnowledgeBaseStats(BaseModel):
    total_profiles: int
    seniority_distribution: dict[str, int]
    avg_experience_years: float
    last_added_at: str | None
    top_skills: list[str]


class KnowledgeBaseResponse(BaseModel):
    stats: KnowledgeBaseStats
    candidates: list[CandidateRecord]
