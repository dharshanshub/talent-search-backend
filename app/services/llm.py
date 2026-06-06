from __future__ import annotations

import logging

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

from app.core.exceptions import UpstreamServiceError

logger = structlog.get_logger(__name__)
_std_logger = logging.getLogger(__name__)  # tenacity requires stdlib logger


class OpenAILLM:
    """Thin wrapper around the OpenAI chat-completions endpoint.

    Retries on transient API errors (rate limits, timeouts, 5xx) up to 3 times
    with exponential back-off. Raises UpstreamServiceError after all retries fail.
    """

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    @retry(
        retry=retry_if_exception_type((APIError, APITimeoutError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(_std_logger, logging.WARNING),
        after=after_log(_std_logger, logging.DEBUG),
        reraise=False,  # we catch the final exception ourselves
    )
    async def _complete_raw(self, system: str, user: str) -> str:
        """Internal: call the chat-completions API. Retried by tenacity on transient errors."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    async def complete(self, system: str, user: str) -> str:
        """Run a chat-completion and return the assistant's text.

        Raises:
            UpstreamServiceError: if all retries are exhausted or an unexpected error occurs.
        """
        try:
            content = await self._complete_raw(system, user)
        except (APIError, APITimeoutError, RateLimitError) as exc:
            logger.error(
                "llm_failed_all_retries",
                model=self._model,
                error=str(exc),
            )
            raise UpstreamServiceError("openai", f"LLM call failed after retries: {exc}") from exc
        except Exception as exc:
            logger.error("llm_unexpected_error", model=self._model, error=str(exc))
            raise UpstreamServiceError("openai", f"LLM error: {exc}") from exc

        logger.debug("llm_complete", model=self._model, output_chars=len(content))
        return content
