from __future__ import annotations

import structlog
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)


class OpenAIEmbedder:
    """Embeds text using OpenAI text-embedding-3-small (1536 dims)."""

    def __init__(self, client: AsyncOpenAI, model: str, dimensions: int) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            input=text,
            model=self._model,
            dimensions=self._dimensions,
        )
        vec = response.data[0].embedding
        logger.debug("embedded_query", dim=len(vec), model=self._model,
                     text_preview=text[:60])
        return vec

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(
            input=texts,
            model=self._model,
            dimensions=self._dimensions,
        )
        sorted_data = sorted(response.data, key=lambda x: x.index)
        logger.debug("embedded_batch", count=len(texts), model=self._model,
                     dim=len(sorted_data[0].embedding) if sorted_data else 0)
        return [item.embedding for item in sorted_data]
