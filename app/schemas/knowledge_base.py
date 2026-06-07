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
    # Experience bucketed into hiring-friendly ranges e.g. {"0–2 yrs": 12, "3–5 yrs": 34}
    experience_distribution: dict[str, int]
    # Top candidate locations and industries for sourcing decisions
    top_locations: list[str]
    top_industries: list[str]
    # True when pool > 10 000 and stats are based on a sample
    is_sampled: bool = False


class CandidatesPageResponse(BaseModel):
    candidates: list[CandidateRecord]
    next_cursor: str | None
    # Total profile count from stats cache; -1 = not yet computed
    total: int


class KnowledgeBaseResponse(BaseModel):
    stats: KnowledgeBaseStats
    candidates: list[CandidateRecord]
