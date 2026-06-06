from __future__ import annotations

import structlog
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    after_log,
)
import logging

from app.core.exceptions import UpstreamServiceError

logger = structlog.get_logger(__name__)
_std_logger = logging.getLogger(__name__)  # tenacity requires stdlib logger


class OpenAIEmbedder:
    """Embeds text using OpenAI text-embedding-3-small (1536 dims).

    Retries on transient API errors (rate limits, timeouts, 5xx) up to 3 times
    with exponential back-off. Raises UpstreamServiceError after all retries fail.
    """

    def __init__(self, client: AsyncOpenAI, model: str, dimensions: int) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions

    @retry(
        retry=retry_if_exception_type((APIError, APITimeoutError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(_std_logger, logging.WARNING),
        after=after_log(_std_logger, logging.DEBUG),
        reraise=False,  # we handle the final exception ourselves
    )
    async def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        """Internal: call OpenAI Embeddings API. Retried by tenacity on transient errors."""
        response = await self._client.embeddings.create(
            input=texts,
            model=self._model,
            dimensions=self._dimensions,
        )
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    async def embed(self, text: str) -> list[float]:
        """Embed a single string.

        Raises:
            UpstreamServiceError: if all retries are exhausted.
        """
        if not text.strip():
            raise ValueError("Cannot embed an empty string")

        try:
            vecs = await self._embed_raw([text])
        except (APIError, APITimeoutError, RateLimitError) as exc:
            logger.error(
                "embed_failed_all_retries",
                model=self._model,
                error=str(exc),
            )
            raise UpstreamServiceError("openai", f"Embedding failed after retries: {exc}") from exc
        except Exception as exc:
            logger.error("embed_unexpected_error", model=self._model, error=str(exc))
            raise UpstreamServiceError("openai", f"Embedding error: {exc}") from exc

        vec = vecs[0]
        logger.debug("embedded_query", dim=len(vec), model=self._model, text_preview=text[:60])
        return vec

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings, preserving order.

        Raises:
            UpstreamServiceError: if all retries are exhausted.
        """
        if not texts:
            return []

        try:
            vecs = await self._embed_raw(texts)
        except (APIError, APITimeoutError, RateLimitError) as exc:
            logger.error(
                "embed_batch_failed_all_retries",
                count=len(texts),
                model=self._model,
                error=str(exc),
            )
            raise UpstreamServiceError(
                "openai", f"Batch embedding failed after retries: {exc}"
            ) from exc
        except Exception as exc:
            logger.error(
                "embed_batch_unexpected_error",
                count=len(texts),
                model=self._model,
                error=str(exc),
            )
            raise UpstreamServiceError("openai", f"Batch embedding error: {exc}") from exc

        logger.debug(
            "embedded_batch",
            count=len(texts),
            model=self._model,
            dim=len(vecs[0]) if vecs else 0,
        )
        return vecs
