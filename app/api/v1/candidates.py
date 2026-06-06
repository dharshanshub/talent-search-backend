from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.core.exceptions import NotFoundError

router = APIRouter(prefix="/candidates", tags=["candidates"])

_RESUME_DIR = Path(__file__).resolve().parents[3] / "data" / "resumes"


@router.get("/{candidate_id}/resume")
async def get_resume(candidate_id: str) -> FileResponse:
    pdf_path = _RESUME_DIR / f"{candidate_id}.pdf"
    if not pdf_path.exists():
        raise NotFoundError(f"Resume not found for candidate '{candidate_id}'")
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"inline; filename={candidate_id}_resume.pdf",
            "Cache-Control": "public, max-age=3600",
        },
    )
