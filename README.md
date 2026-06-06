# Backend — Talent Search RAG API

FastAPI service exposing the RAG pipeline.

## Local dev (without Docker)

```bash
cp .env.example .env   # fill in keys
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:create_app --factory --reload
```

## Endpoints

| Method | Path             | Description              |
|--------|------------------|--------------------------|
| GET    | /healthz         | Liveness probe           |
| GET    | /readyz          | Readiness probe          |
| POST   | /api/v1/search   | Talent search            |
| GET    | /docs            | Swagger UI (dev only)    |

## Tests

```bash
pytest app/tests -v
```

## Lint / format

```bash
ruff check app/
ruff format app/
```
