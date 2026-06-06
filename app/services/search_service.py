from __future__ import annotations

from datetime import date

import structlog

from app.schemas.search import CandidateMatch, SearchRequest, SearchResponse
from app.services.embeddings import OpenAIEmbedder
from app.services.llm import OpenAILLM
from app.services.query_understanding import QueryUnderstandingService
from app.services.vector_store import PineconeStore

logger = structlog.get_logger(__name__)

_ANSWER_SYSTEM = """You are a talent search assistant helping recruiters find the right candidates.

Given a recruiter's search query and a ranked list of matching candidate profiles, write a concise 2-3 sentence summary of the best matches.

Rules:
- Mention the top 2-3 candidates by name
- Highlight their most relevant skills and experience for the query
- Note how recently each profile was updated
- Be professional, specific, and helpful
- Do NOT invent information beyond what is provided
"""


class SearchService:
    def __init__(
        self,
        embedder: OpenAIEmbedder,
        vector_store: PineconeStore,
        query_understanding: QueryUnderstandingService,
        llm: OpenAILLM,
        top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._query_understanding = query_understanding
        self._llm = llm
        self._top_k = top_k

    async def search(self, request: SearchRequest, request_id: str) -> SearchResponse:
        top_k = request.top_k or self._top_k

        # ── 1. Parse NL query into structured fields ───────────────────────
        parsed = await self._query_understanding.parse(request.query)

        # ── 2. Embed the semantic text ─────────────────────────────────────
        vector = await self._embedder.embed(parsed.semantic_text)

        # ── 3. Build optional Pinecone metadata filter ─────────────────────
        pinecone_filter: dict | None = None
        if parsed.min_years:
            pinecone_filter = {"years_experience": {"$gte": parsed.min_years}}

        logger.info(
            "search_pinecone_query",
            semantic_text=parsed.semantic_text,
            vector_dim=len(vector),
            filter=pinecone_filter,
            top_k_requested=top_k * 5,
        )

        # Fetch more than top_k to survive deduplication
        raw_matches = await self._store.query(
            vector=vector,
            top_k=top_k * 5,
            filter=pinecone_filter,
        )

        # ── 4. Deduplicate — keep best-scoring chunk per candidate ─────────
        best: dict[str, dict] = {}
        for match in raw_matches:
            cid = match["metadata"].get("candidate_id", match["id"])
            if cid not in best or match["score"] > best[cid]["score"]:
                best[cid] = match

        top_matches = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:top_k]

        logger.info(
            "retrieval_done",
            raw_hits=len(raw_matches),
            unique_candidates=len(best),
            returned=len(top_matches),
        )

        # ── 5. Build CandidateMatch objects ────────────────────────────────
        candidates: list[CandidateMatch] = []
        for match in top_matches:
            meta = match["metadata"]
            skills_raw = meta.get("skills", "")
            skills_list = [s.strip() for s in skills_raw.split(",") if s.strip()]

            try:
                last_updated = date.fromisoformat(meta.get("last_updated", "2024-01-01"))
            except ValueError:
                last_updated = date(2024, 1, 1)

            candidates.append(
                CandidateMatch(
                    id=meta.get("candidate_id", match["id"]),
                    name=meta.get("name", "Unknown"),
                    title=meta.get("title", ""),
                    location=meta.get("location", ""),
                    skills=skills_list,
                    years_experience=int(meta.get("years_experience", 0)),
                    last_updated=last_updated,
                    score=round(float(match["score"]), 4),
                )
            )

        # ── 6. Generate natural-language answer ────────────────────────────
        answer = await self._generate_answer(request.query, candidates)

        return SearchResponse(
            query=request.query,
            answer=answer,
            candidates=candidates,
            request_id=request_id,
        )

    async def _generate_answer(self, query: str, candidates: list[CandidateMatch]) -> str:
        if not candidates:
            return (
                "No matching candidates were found for your query. "
                "Try broadening the search or removing specific filters."
            )

        candidate_lines = "\n".join(
            f"{i+1}. {c.name} | {c.title} | {c.location} | "
            f"Skills: {', '.join(c.skills[:6])} | "
            f"{c.years_experience} yrs exp | "
            f"Last updated: {c.last_updated} | "
            f"Match score: {c.score:.0%}"
            for i, c in enumerate(candidates)
        )

        user_prompt = (
            f"Recruiter query: {query}\n\n"
            f"Top matching candidates:\n{candidate_lines}"
        )

        return await self._llm.complete(system=_ANSWER_SYSTEM, user=user_prompt)
