from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"
    # Comma-separated string to avoid pydantic-settings JSON-parsing the env var
    cors_origins: str = "http://localhost:5173"

    # OpenAI
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-4o-mini"

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # RAG
    embedding_dim: int = 1536
    top_k: int = 5

    # Azure Blob Storage — optional; leave empty to fall back to local disk
    azure_storage_connection_string: str = ""
    azure_storage_container: str = "resumes"

    # JWT — set JWT_SECRET_KEY to a random 32+ char string in production
    jwt_secret_key: str = "change-me-in-production-use-a-long-random-string"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 480  # 8 hours

    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
