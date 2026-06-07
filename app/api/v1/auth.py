from __future__ import annotations

import json

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from app.core.logging import get_correlation_id
from app.core.security import create_access_token, verify_password

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

_USERS_BLOB = "users.json"
_INVALID_CREDENTIALS = "Invalid username or password"

# Dummy hash used in constant-time comparison when username is not found,
# preventing timing attacks that could reveal valid usernames.
_DUMMY_HASH = "$2b$12$aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa2"


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


async def _load_users(request: Request) -> list[dict]:
    """Load users from Blob Storage. Cached in app.state for the process lifetime.

    The cache is intentionally never invalidated during a run — a restart is
    required to pick up credential changes (acceptable for a single-user setup).
    """
    if hasattr(request.app.state, "auth_users_cache"):
        return request.app.state.auth_users_cache  # type: ignore[no-any-return]

    blob_service = request.app.state.blob_service
    if not blob_service.available:
        logger.error("auth_blob_unavailable", hint="Set AZURE_STORAGE_CONNECTION_STRING")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable",
        )

    try:
        raw = await blob_service.download(_USERS_BLOB)
        data = json.loads(raw)
        users: list[dict] = data.get("users", [])
        request.app.state.auth_users_cache = users
        logger.info("auth_users_loaded", count=len(users))
        return users
    except Exception as exc:
        logger.error("auth_users_load_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable",
        ) from exc


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, request: Request) -> TokenResponse:
    """Authenticate with username + password, receive a JWT bearer token.

    The token is valid for 8 hours.  Include it on every subsequent request as:
        Authorization: Bearer <token>
    """
    request_id = get_correlation_id()
    logger.info("login_attempt", username=body.username, request_id=request_id)

    users = await _load_users(request)
    user = next((u for u in users if u.get("username") == body.username), None)

    # Always run bcrypt to prevent username-enumeration via timing
    candidate_hash = user["password_hash"] if user else _DUMMY_HASH
    password_ok = verify_password(body.password, candidate_hash)

    if not user or not password_ok:
        logger.warning("login_failed", username=body.username, request_id=request_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_INVALID_CREDENTIALS,
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(username=body.username)
    logger.info("login_success", username=body.username, request_id=request_id)
    return TokenResponse(access_token=token)
