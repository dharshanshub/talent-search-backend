"""
Phase 2b -- Reads 100 resume PDFs, chunks them, embeds with OpenAI,
and upserts into Pinecone with rich metadata.

Idempotent: re-running overwrites vectors by the same deterministic IDs.

Usage:
    python -m app.scripts.seed_index
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI
from pypdf import PdfReader
from tenacity import retry, stop_after_attempt, wait_exponential

# Allow running as __main__ from the backend root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.services.embeddings import OpenAIEmbedder
from app.services.vector_store import PineconeStore

logger = structlog.get_logger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).resolve().parents[2] / "data"
RESUME_DIR  = DATA_DIR / "resumes"
PROFILES_JSON = DATA_DIR / "profiles.json"

# ── chunking config ───────────────────────────────────────────────────────────
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""],
)


# ── PDF text extraction ───────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


# ── chunk type inference ──────────────────────────────────────────────────────

def _infer_chunk_type(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("summary", "objective", "profile")):
        return "summary"
    if any(k in t for k in ("skill", "technology", "framework", "language")):
        return "skills"
    if any(k in t for k in ("experience", "engineer", "developer", "analyst", "scientist")):
        return "experience"
    if any(k in t for k in ("education", "university", "degree", "bachelor", "master")):
        return "education"
    return "general"


# ── metadata builder ──────────────────────────────────────────────────────────

def build_vectors(
    candidate_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Pairs each chunk + embedding with Pinecone-ready metadata."""

    skills_str     = ",".join(profile.get("skills", []))
    industries_set = {
        exp.get("industry", "")
        for exp in profile.get("experience", [])
        if exp.get("industry")
    }
    industries_str = ",".join(sorted(industries_set))

    # Extract role and seniority from title e.g. "Senior Backend Engineer"
    title      = profile.get("title", "")
    title_parts = title.split()
    seniority_words = {"Junior", "Mid-Level", "Senior", "Staff", "Principal"}
    seniority  = title_parts[0] if title_parts[0] in seniority_words else "Mid-Level"
    role       = " ".join(
        w for w in title_parts if w not in seniority_words
    ).strip() or title

    vectors = []
    for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"{candidate_id}_chunk_{idx}",
            "values": embedding,
            "metadata": {
                # chunk context
                "text":          chunk_text,
                "candidate_id":  candidate_id,
                "chunk_index":   idx,
                "chunk_type":    _infer_chunk_type(chunk_text),
                # candidate fields (filterable)
                "name":               profile.get("name", ""),
                "title":              title,
                "role":               role,
                "seniority":          seniority,
                "location":           profile.get("location", ""),
                "years_experience":   profile.get("years_experience", 0),
                "skills":             skills_str,
                "industries":         industries_str,
                "last_updated":       profile.get("last_updated", ""),
            },
        })
    return vectors


# ── Pinecone index setup ──────────────────────────────────────────────────────

def ensure_index(settings: Any) -> Any:
    """Creates the Pinecone serverless index if it doesn't exist, then returns it."""
    from pinecone import Pinecone, ServerlessSpec  # type: ignore[import]

    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing_names = [idx.name for idx in pc.list_indexes()]

    if settings.pinecone_index_name not in existing_names:
        logger.info(
            "creating_pinecone_index",
            name=settings.pinecone_index_name,
            dim=settings.embedding_dim,
        )
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        # Wait until index is ready
        logger.info("waiting_for_index_ready")
        while not pc.describe_index(settings.pinecone_index_name).status.ready:
            time.sleep(2)
        logger.info("index_ready")
    else:
        logger.info("index_exists", name=settings.pinecone_index_name)

    return pc.Index(settings.pinecone_index_name)


# ── main pipeline ─────────────────────────────────────────────────────────────

async def seed() -> None:
    settings = get_settings()
    configure_logging(app_env=settings.app_env, log_level=settings.log_level)

    # Validate inputs
    if not PROFILES_JSON.exists():
        raise FileNotFoundError(f"Run generate_resumes.py first. Missing: {PROFILES_JSON}")
    if not RESUME_DIR.exists():
        raise FileNotFoundError(f"Resume PDFs missing at: {RESUME_DIR}")

    profiles: list[dict[str, Any]] = json.loads(PROFILES_JSON.read_text(encoding="utf-8"))
    logger.info("loaded_profiles", count=len(profiles))

    # Init OpenAI
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    embedder = OpenAIEmbedder(
        client=openai_client,
        model=settings.openai_embedding_model,
        dimensions=settings.embedding_dim,
    )

    # Init Pinecone
    logger.info("connecting_pinecone")
    index = ensure_index(settings)
    store = PineconeStore(index=index)
    logger.info("pinecone_ready")

    total_vectors = 0
    failed: list[str] = []

    for i, profile in enumerate(profiles, 1):
        candidate_id = profile["candidate_id"]
        pdf_path     = RESUME_DIR / f"{candidate_id}.pdf"

        if not pdf_path.exists():
            logger.warning("pdf_missing", candidate_id=candidate_id)
            failed.append(candidate_id)
            continue

        try:
            # 1. Extract text from PDF
            raw_text = extract_pdf_text(pdf_path)
            if not raw_text:
                logger.warning("empty_pdf_text", candidate_id=candidate_id)
                failed.append(candidate_id)
                continue

            # 2. Chunk
            chunks = splitter.split_text(raw_text)
            if not chunks:
                logger.warning("no_chunks", candidate_id=candidate_id)
                continue

            # 3. Embed all chunks in one batch call
            embeddings = await embedder.embed_batch(chunks)

            # 4. Build vectors with metadata
            vectors = build_vectors(candidate_id, chunks, embeddings, profile)

            # 5. Upsert to Pinecone
            await store.upsert(vectors)

            total_vectors += len(vectors)
            logger.info(
                "indexed",
                candidate_id=candidate_id,
                chunks=len(chunks),
                progress=f"{i}/{len(profiles)}",
            )

        except Exception as exc:
            logger.error("indexing_failed", candidate_id=candidate_id, error=str(exc))
            failed.append(candidate_id)

    await openai_client.close()

    # Summary
    print(f"\n{'='*50}")
    print(f"Indexing complete.")
    print(f"  Candidates processed : {len(profiles) - len(failed)}/{len(profiles)}")
    print(f"  Total vectors upserted: {total_vectors}")
    print(f"  Failed               : {len(failed)}")
    if failed:
        print(f"  Failed IDs           : {', '.join(failed)}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Load .env manually when running as script (not via uvicorn)
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    asyncio.run(seed())
