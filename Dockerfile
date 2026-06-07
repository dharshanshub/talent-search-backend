# syntax=docker/dockerfile:1
# ── stage 1: dependency installer ────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install into an isolated venv so the runtime stage stays clean
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install deps first (layer-cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --no-create-home appuser

WORKDIR /app

# Carry over only the built venv — no build tools in final image
COPY --from=builder /opt/venv /opt/venv

# Copy application source
COPY --chown=appuser:appgroup app/ app/

# /app/data is mounted as a volume at runtime (resumes + profiles.json)
# Create the mount-point directory with correct ownership
RUN mkdir -p /app/data && chown appuser:appgroup /app/data

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
