from __future__ import annotations

import io
from pathlib import Path

import structlog
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import FileResponse
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.exceptions import BadRequestError, NotFoundError, UpstreamServiceError
from app.core.logging import get_correlation_id
from app.schemas.candidate import IndexRequest, IndexResponse, UploadResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/candidates", tags=["candidates"])

_RESUME_DIR = Path(__file__).resolve().parents[3] / "data" / "resumes"
_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


# ── Serve an existing resume PDF ──────────────────────────────────────────────

@router.get("/{candidate_id}/resume")
async def get_resume(candidate_id: str, request: Request) -> FileResponse:
    """Serve the stored PDF for a known candidate.

    Raises:
        NotFoundError: if no PDF exists for the given candidate_id.
    """
    request_id = get_correlation_id()

    # Basic path-traversal guard — candidate IDs are alphanumeric + underscores
    if not candidate_id.replace("_", "").replace("-", "").isalnum():
        logger.warning(
            "resume_invalid_id",
            candidate_id=candidate_id,
            request_id=request_id,
        )
        raise BadRequestError("Invalid candidate ID")

    pdf_path = _RESUME_DIR / f"{candidate_id}.pdf"
    if not pdf_path.exists():
        logger.info(
            "resume_not_found",
            candidate_id=candidate_id,
            path=str(pdf_path),
            request_id=request_id,
        )
        raise NotFoundError(f"Resume not found for candidate '{candidate_id}'")

    logger.info("resume_served", candidate_id=candidate_id, request_id=request_id)
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename={candidate_id}_resume.pdf",
            "Cache-Control": "public, max-age=3600",
        },
    )


# ── Upload PDF → LLM extraction ───────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_resume(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    """Accept a resume PDF, extract text, run LLM extraction, return structured profile.

    Raises:
        BadRequestError: for non-PDF files, oversized files, unreadable PDFs, or empty text.
        UpstreamServiceError: if the OpenAI extraction call fails.
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

    # Read once; avoid streaming edge-cases
    try:
        contents = await file.read()
    except Exception as exc:
        logger.error("upload_read_failed", error=str(exc), request_id=request_id)
        raise BadRequestError(f"Could not read uploaded file: {exc}") from exc

    if len(contents) > _MAX_FILE_BYTES:
        raise BadRequestError(
            f"File too large — maximum {_MAX_FILE_BYTES // (1024 * 1024)} MB"
        )
    if len(contents) == 0:
        raise BadRequestError("Uploaded file is empty")

    logger.info(
        "upload_received",
        filename=file.filename,
        bytes=len(contents),
        request_id=request_id,
    )

    # Extract text from PDF bytes
    try:
        reader = PdfReader(io.BytesIO(contents))
        pages = [page.extract_text() or "" for page in reader.pages]
        raw_text = "\n\n".join(pages).strip()
    except PdfReadError as exc:
        logger.error(
            "pdf_corrupt",
            error=str(exc),
            filename=file.filename,
            request_id=request_id,
        )
        raise BadRequestError(f"PDF file is corrupted or password-protected: {exc}") from exc
    except Exception as exc:
        logger.error(
            "pdf_parse_failed",
            error=str(exc),
            filename=file.filename,
            request_id=request_id,
        )
        raise BadRequestError(f"Could not parse PDF: {exc}") from exc

    if not raw_text:
        raise BadRequestError(
            "PDF appears to contain no extractable text — "
            "it may be a scanned image. Please use a text-based PDF."
        )

    logger.info(
        "pdf_parsed",
        filename=file.filename,
        chars=len(raw_text),
        pages=len(pages),
        request_id=request_id,
    )

    # LLM extraction — errors are typed (BadRequestError or UpstreamServiceError)
    screening = request.app.state.screening_service
    try:
        extracted = await screening.extract(raw_text)
    except (BadRequestError, UpstreamServiceError):
        raise  # already structured and logged
    except Exception as exc:
        logger.error("extraction_unexpected", error=str(exc), request_id=request_id)
        raise UpstreamServiceError("openai", f"Unexpected extraction error: {exc}") from exc

    return UploadResponse(extracted=extracted, raw_text=raw_text)


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
        name=body.profile.name,
        title=body.profile.title,
        raw_text_chars=len(body.raw_text),
        request_id=request_id,
    )

    screening = request.app.state.screening_service
    try:
        result = await screening.index(body.profile, body.raw_text)
    except (BadRequestError, UpstreamServiceError):
        raise  # already structured and logged
    except Exception as exc:
        logger.error("index_unexpected", error=str(exc), request_id=request_id)
        raise UpstreamServiceError("pinecone", f"Unexpected indexing error: {exc}") from exc

    logger.info(
        "index_done",
        candidate_id=result.candidate_id,
        chunks=result.chunks_indexed,
        name=body.profile.name,
        request_id=request_id,
    )
    return result
