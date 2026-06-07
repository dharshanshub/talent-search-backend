from __future__ import annotations

import io
import uuid
from pathlib import Path

import structlog
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.exceptions import BadRequestError, NotFoundError, UpstreamServiceError
from app.core.logging import get_correlation_id
from app.schemas.candidate import IndexRequest, IndexResponse, UploadResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/candidates", tags=["candidates"])

_RESUME_DIR = Path(__file__).resolve().parents[3] / "data" / "resumes"
_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


# ── Serve a resume PDF ────────────────────────────────────────────────────────

@router.get("/{candidate_id}/resume", response_model=None)
async def get_resume(candidate_id: str, request: Request) -> FileResponse | RedirectResponse:
    """Serve a candidate's resume PDF.

    Priority:
      1. Azure Blob Storage → 302 redirect to a 60-min SAS URL (when configured)
      2. Local disk fallback (dev / legacy seeded candidates)
      3. 404 if neither source has the file

    Raises:
        BadRequestError: if the candidate_id contains path-traversal characters.
        NotFoundError: if the PDF cannot be found in blob or on disk.
    """
    request_id = get_correlation_id()

    # Guard against path traversal — candidate IDs are alphanumeric + underscores/hyphens
    if not all(c.isalnum() or c in "_-" for c in candidate_id):
        logger.warning("resume_invalid_id", candidate_id=candidate_id, request_id=request_id)
        raise BadRequestError("Invalid candidate ID")

    blob_name = f"{candidate_id}.pdf"
    blob_service = request.app.state.blob_service

    # 1 — Try Blob Storage (Azure)
    if blob_service.available:
        try:
            sas_url = await blob_service.get_sas_url(blob_name, expiry_minutes=60)
            if sas_url:
                logger.info("resume_served_blob", candidate_id=candidate_id, request_id=request_id)
                return RedirectResponse(url=sas_url, status_code=302)
        except UpstreamServiceError as exc:
            # Log but fall through to local disk — don't hard-fail for a SAS error
            logger.warning("resume_blob_fallback", reason=str(exc), candidate_id=candidate_id)

    # 2 — Fall back to local disk (dev environment / legacy seeded candidates)
    pdf_path = _RESUME_DIR / blob_name
    if pdf_path.exists():
        logger.info("resume_served_local", candidate_id=candidate_id, request_id=request_id)
        return FileResponse(
            str(pdf_path),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"inline; filename={blob_name}",
                "Cache-Control": "public, max-age=3600",
            },
        )

    logger.info("resume_not_found", candidate_id=candidate_id, request_id=request_id)
    raise NotFoundError(f"Resume not found for candidate '{candidate_id}'")


# ── Upload PDF → LLM extraction ───────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_resume(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    """Accept a resume PDF, save it to Blob Storage, run LLM extraction.

    The candidate_id is generated here (not at index time) so the PDF is already
    named and stored before the user reaches the human-review step.

    Raises:
        BadRequestError: for wrong type, oversized file, unreadable PDF, or empty text.
        UpstreamServiceError: if Blob upload or OpenAI extraction fails.
    """
    request_id = get_correlation_id()

    if file.content_type not in ("application/pdf", "application/octet-stream"):
        logger.warning(
            "upload_wrong_content_type",
            content_type=file.content_type,
            filename=file.filename,
            request_id=request_id,
        )
        raise BadRequestError("Only PDF files are supported")

    try:
        contents = await file.read()
    except Exception as exc:
        logger.error("upload_read_failed", error=str(exc), request_id=request_id)
        raise BadRequestError(f"Could not read uploaded file: {exc}") from exc

    if len(contents) == 0:
        raise BadRequestError("Uploaded file is empty")
    if len(contents) > _MAX_FILE_BYTES:
        raise BadRequestError(f"File too large — maximum {_MAX_FILE_BYTES // (1024 * 1024)} MB")

    # Generate candidate_id now so blob and Pinecone share the same key
    candidate_id = f"uploaded_{uuid.uuid4().hex[:10]}"
    blob_filename = f"{candidate_id}.pdf"

    logger.info(
        "upload_received",
        filename=file.filename,
        bytes=len(contents),
        candidate_id=candidate_id,
        request_id=request_id,
    )

    # Save PDF to Blob Storage (no-op locally when Azure is not configured)
    blob_service = request.app.state.blob_service
    try:
        await blob_service.upload(blob_filename, contents)
    except UpstreamServiceError:
        raise
    except Exception as exc:
        logger.error("upload_blob_failed", error=str(exc), request_id=request_id)
        raise UpstreamServiceError("azure_blob", f"Blob upload failed: {exc}") from exc

    # Extract text from PDF bytes
    try:
        reader = PdfReader(io.BytesIO(contents))
        pages = [page.extract_text() or "" for page in reader.pages]
        raw_text = "\n\n".join(pages).strip()
    except PdfReadError as exc:
        logger.error("pdf_corrupt", error=str(exc), filename=file.filename, request_id=request_id)
        raise BadRequestError(f"PDF is corrupted or password-protected: {exc}") from exc
    except Exception as exc:
        logger.error("pdf_parse_failed", error=str(exc), filename=file.filename, request_id=request_id)
        raise BadRequestError(f"Could not parse PDF: {exc}") from exc

    if not raw_text:
        raise BadRequestError(
            "PDF contains no extractable text — it may be a scanned image. "
            "Please use a text-based PDF."
        )

    logger.info(
        "pdf_parsed",
        filename=file.filename,
        chars=len(raw_text),
        pages=len(pages),
        request_id=request_id,
    )

    # LLM extraction
    screening = request.app.state.screening_service
    try:
        extracted = await screening.extract(raw_text)
    except (BadRequestError, UpstreamServiceError):
        raise
    except Exception as exc:
        logger.error("extraction_unexpected", error=str(exc), request_id=request_id)
        raise UpstreamServiceError("openai", f"Unexpected extraction error: {exc}") from exc

    return UploadResponse(
        candidate_id=candidate_id,
        blob_filename=blob_filename,
        extracted=extracted,
        raw_text=raw_text,
    )


# ── Index validated candidate ─────────────────────────────────────────────────

@router.post("/index", response_model=IndexResponse)
async def index_candidate(body: IndexRequest, request: Request) -> IndexResponse:
    """Chunk, embed, and upsert a human-validated candidate profile into Pinecone.

    Raises:
        BadRequestError: if raw_text is empty or produces no chunks.
        UpstreamServiceError: if OpenAI embedding or Pinecone upsert fails.
    """
    request_id = get_correlation_id()

    if not body.raw_text.strip():
        raise BadRequestError("raw_text must not be empty")

    logger.info(
        "index_request",
        candidate_id=body.candidate_id,
        blob_filename=body.blob_filename,
        name=body.profile.name,
        request_id=request_id,
    )

    screening = request.app.state.screening_service
    try:
        result = await screening.index(
            candidate_id=body.candidate_id,
            blob_filename=body.blob_filename,
            profile=body.profile,
            raw_text=body.raw_text,
        )
    except (BadRequestError, UpstreamServiceError):
        raise
    except Exception as exc:
        logger.error("index_unexpected", error=str(exc), request_id=request_id)
        raise UpstreamServiceError("pinecone", f"Unexpected indexing error: {exc}") from exc

    # Invalidate KB stats cache — total_profiles has increased
    if hasattr(request.app.state, "kb_stats_cache"):
        request.app.state.kb_stats_cache["cached_at"] = None

    logger.info(
        "index_done",
        candidate_id=result.candidate_id,
        chunks=result.chunks_indexed,
        name=body.profile.name,
        request_id=request_id,
    )
    return result
