from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# Provide minimal env vars so Settings can instantiate without a real .env
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
os.environ.setdefault("PINECONE_API_KEY", "test-placeholder")
os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")

from app.main import create_app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
