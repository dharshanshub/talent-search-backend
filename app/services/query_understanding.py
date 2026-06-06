from __future__ import annotations

import json
import re

import structlog
from pydantic import BaseModel

from app.services.llm import OpenAILLM

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are a talent search query parser.

Extract structured information from a recruiter's natural-language query and return ONLY a valid JSON object — no markdown, no explanation.

JSON fields:
- semantic_text  : string  — clean version of the query for semantic embedding
- skills         : array   — specific technologies / frameworks / tools mentioned
- location       : string | null — city or country if mentioned, else null
- min_years      : number | null — minimum years of experience if mentioned, else null

Examples:
Query: "senior React engineer in Berlin with 5+ years fintech experience"
{"semantic_text":"senior React engineer Berlin fintech","skills":["React"],"location":"Berlin","min_years":5}

Query: "ML engineer who knows PyTorch and has worked in healthcare"
{"semantic_text":"machine learning engineer PyTorch healthcare","skills":["PyTorch"],"location":null,"min_years":null}

Query: "junior backend developer Python Django"
{"semantic_text":"junior backend developer Python Django","skills":["Python","Django"],"location":null,"min_years":null}
"""


class ParsedQuery(BaseModel):
    semantic_text: str
    skills: list[str] = []
    location: str | None = None
    min_years: int | None = None


class QueryUnderstandingService:
    def __init__(self, llm_client: OpenAILLM) -> None:
        self._llm = llm_client

    async def parse(self, query: str) -> ParsedQuery:
        raw = await self._llm.complete(system=_SYSTEM_PROMPT, user=query)

        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

        try:
            data = json.loads(raw)
            parsed = ParsedQuery(**data)
        except Exception as exc:
            logger.warning(
                "query_parse_fallback",
                error=str(exc),
                raw=raw[:200],
            )
            # Graceful fallback: use the original query as semantic text
            parsed = ParsedQuery(semantic_text=query)

        logger.info(
            "query_parsed",
            semantic_text=parsed.semantic_text,
            skills=parsed.skills,
            location=parsed.location,
            min_years=parsed.min_years,
        )
        return parsed
