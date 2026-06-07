from __future__ import annotations

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
    skills: list[str] = Field(default_factory=list, description="Technical skills only")
    industries: list[str] = Field(default_factory=list, description="Industries worked in")
    summary: str = Field(..., description="2-3 sentence professional summary")


class UploadResponse(BaseModel):
    """Returned after PDF upload + LLM extraction.

    candidate_id and blob_filename are generated at upload time so the PDF is
    already named and stored before the user reaches the review step.
    """
    candidate_id: str = Field(..., description="Generated ID — used as Pinecone + blob key")
    blob_filename: str = Field(..., description="Blob name e.g. 'uploaded_abc123.pdf'")
    extracted: ExtractedProfile
    raw_text: str


class IndexRequest(BaseModel):
    """Sent by the frontend after the user reviews and confirms the extracted profile."""
    candidate_id: str = Field(..., description="ID assigned at upload time")
    blob_filename: str = Field(..., description="Blob filename assigned at upload time")
    profile: ExtractedProfile
    raw_text: str


class IndexResponse(BaseModel):
    candidate_id: str
    chunks_indexed: int
    message: str
