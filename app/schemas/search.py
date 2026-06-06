from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Natural-language talent search query")
    top_k: int | None = Field(None, ge=1, le=20, description="Override default result count")


class CandidateMatch(BaseModel):
    id: str
    name: str
    title: str
    location: str
    skills: list[str]
    years_experience: int
    last_updated: date
    score: float = Field(..., description="Similarity score [0, 1]")
    summary: str | None = None


class SearchResponse(BaseModel):
    query: str
    answer: str
    candidates: list[CandidateMatch]
    request_id: str
