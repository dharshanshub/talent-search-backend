from __future__ import annotations

import json
import re
import uuid
from datetime import date
from typing import Any

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ValidationError

from app.core.exceptions import BadRequestError, UpstreamServiceError
from app.schemas.candidate import ExtractedProfile, IndexResponse
from app.services.embeddings import OpenAIEmbedder
from app.services.llm import OpenAILLM
from app.services.vector_store import PineconeStore

logger = structlog.get_logger(__name__)

_EXTRACT_SYSTEM = """You are an expert resume parser. Extract structured information from the resume text below.

Return ONLY a valid JSON object with exactly these fields — no markdown, no explanation:

{
  "name": "Full name of the candidate",
  "title": "Most recent or current job title (full, e.g. 'Senior Backend Engineer')",
  "role": "Role without seniority prefix (e.g. 'Backend Engineer')",
  "seniority": "One of: Junior, Mid-Level, Senior, Staff, Principal",
  "location": "City, Country",
  "years_experience": 5,
  "skills": ["Python", "React", "PostgreSQL"],
  "industries": ["Fintech", "Healthcare"],
  "summary": "2-3 sentence professional summary highlighting key strengths"
}

Rules:
- years_experience must be an integer
- skills: technical skills only, no soft skills
- seniority: infer from title and experience if not explicit
- If a field cannot be determined, use a sensible default (empty string, 0, or empty list)
"""

# Text splitter shared across all calls — thread-safe, stateless
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""],
)

# Truncate resumes at 6000 chars to stay within LLM context limits
_MAX_RESUME_CHARS = 6000


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that the LLM sometimes wraps JSON in."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


class ScreeningService:
    """Orchestrates LLM-based resume extraction and Pinecone indexing."""

    def __init__(
        self,
        llm: OpenAILLM,
        embedder: OpenAIEmbedder,
        vector_store: PineconeStore,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._store = vector_store

    async def extract(self, raw_text: str) -> ExtractedProfile:
        """Send resume text to LLM and parse the response into an ExtractedProfile.

        Raises:
            BadRequestError: if the resume text is empty.
            UpstreamServiceError: if the LLM call fails after retries.
            BadRequestError: if the LLM returns unparseable JSON or a schema mismatch.
        """
        if not raw_text.strip():
            raise BadRequestError("Resume text is empty — nothing to extract")

        logger.info("extraction_start", text_chars=len(raw_text))

        try:
            response = await self._llm.complete(
                system=_EXTRACT_SYSTEM,
                user=f"Resume text:\n\n{raw_text[:_MAX_RESUME_CHARS]}",
            )
        except UpstreamServiceError:
            raise  # already logged and typed
        except Exception as exc:
            logger.error("extraction_llm_failed", error=str(exc))
            raise UpstreamServiceError("openai", f"LLM call failed: {exc}") from exc

        cleaned = _strip_fences(response)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(
                "extraction_json_parse_failed",
                error=str(exc),
                raw_preview=cleaned[:300],
            )
            raise BadRequestError(
                f"LLM returned invalid JSON — cannot parse resume. Detail: {exc}"
            ) from exc

        try:
            profile = ExtractedProfile(**data)
        except ValidationError as exc:
            logger.error(
                "extraction_schema_mismatch",
                errors=exc.errors(),
                raw_preview=cleaned[:300],
            )
            raise BadRequestError(
                f"Extracted data does not match expected schema: {exc.error_count()} field error(s)"
            ) from exc

        logger.info(
            "extraction_done",
            name=profile.name,
            title=profile.title,
            skills_count=len(profile.skills),
            years=profile.years_experience,
        )
        return profile

    async def index(self, profile: ExtractedProfile, raw_text: str) -> IndexResponse:
        """Chunk, embed, and upsert a validated candidate profile into Pinecone.

        Raises:
            BadRequestError: if the resume text produces no chunks.
            UpstreamServiceError: if embedding or upsert fails.
        """
        candidate_id = f"uploaded_{uuid.uuid4().hex[:10]}"
        today = date.today().isoformat()

        logger.info(
            "indexing_start",
            candidate_id=candidate_id,
            name=profile.name,
            raw_text_chars=len(raw_text),
        )

        chunks = _SPLITTER.split_text(raw_text)
        if not chunks:
            raise BadRequestError("Resume text produced no text chunks — cannot index")

        logger.debug("indexing_chunks_created", count=len(chunks), candidate_id=candidate_id)

        try:
            embeddings = await self._embedder.embed_batch(chunks)
        except UpstreamServiceError:
            raise
        except Exception as exc:
            logger.error("indexing_embed_failed", error=str(exc), candidate_id=candidate_id)
            raise UpstreamServiceError("openai", f"Embedding failed: {exc}") from exc

        if len(embeddings) != len(chunks):
            raise UpstreamServiceError(
                "openai",
                f"Embedding count mismatch: expected {len(chunks)}, got {len(embeddings)}",
            )

        skills_str = ",".join(profile.skills)
        industries_str = ",".join(profile.industries)

        vectors: list[dict[str, Any]] = [
            {
                "id": f"{candidate_id}_chunk_{idx}",
                "values": embedding,
                "metadata": {
                    "text":             chunk,
                    "candidate_id":     candidate_id,
                    "chunk_index":      idx,
                    "chunk_type":       "general",
                    "name":             profile.name,
                    "title":            profile.title,
                    "role":             profile.role,
                    "seniority":        profile.seniority,
                    "location":         profile.location,
                    "years_experience": profile.years_experience,
                    "skills":           skills_str,
                    "industries":       industries_str,
                    "last_updated":     today,
                },
            }
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        try:
            await self._store.upsert(vectors)
        except UpstreamServiceError:
            raise
        except Exception as exc:
            logger.error("indexing_upsert_failed", error=str(exc), candidate_id=candidate_id)
            raise UpstreamServiceError("pinecone", f"Upsert failed: {exc}") from exc

        logger.info(
            "indexing_done",
            candidate_id=candidate_id,
            chunks_indexed=len(vectors),
            name=profile.name,
        )
        return IndexResponse(
            candidate_id=candidate_id,
            chunks_indexed=len(vectors),
            message=f"{profile.name} indexed successfully",
        )
