from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


SENIORITY = Literal["Junior", "Mid-Level", "Senior", "Staff", "Principal"]


class ExtractedProfile(BaseModel):
    """LLM-extracted candidate profile — fields match the Pinecone metadata schema."""
    name: str = Field(..., description="Full name")
    title: str = Field(..., description="Full job title e.g. 'Senior Backend Engineer'")
    role: str = Field(..., description="Role without seniority e.g. 'Backend Engineer'")
    seniority: SENIORITY = Field(..., description="Seniority level")
    location: str = Field(..., description="City, Country")
    years_experience: int = Field(..., ge=0, le=50)
    skills: list[str] = Field(default_factory=list, description="Technical skills")
    industries: list[str] = Field(default_factory=list, description="Industries worked in")
    summary: str = Field(..., description="2-3 sentence professional summary")


class UploadResponse(BaseModel):
    extracted: ExtractedProfile
    raw_text: str


class IndexRequest(BaseModel):
    profile: ExtractedProfile
    raw_text: str


class IndexResponse(BaseModel):
    candidate_id: str
    chunks_indexed: int
    message: str
