"""Retry helpers with exponential backoff for API calls.

Provides async and sync retry wrappers for transient HTTP failures
(429, 500, 502, 503, 504) and connection errors.
"""

import asyncio
import logging
import random
import time
from typing import Callable, Set, Tuple, Type

import httpx

logger = logging.getLogger(__name__)

DEFAULT_RETRYABLE_STATUSES: Set[int] = {429, 500, 502, 503, 504}
DEFAULT_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    httpx.TimeoutException,
    httpx.ConnectError,
)


async def retry_with_backoff(
    coro_factory: Callable[[], object],
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    retryable_statuses: Set[int] = DEFAULT_RETRYABLE_STATUSES,
    retryable_exceptions: Tuple[Type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
) -> httpx.Response:
    """Retry an async HTTP call with exponential backoff and jitter.

    Args:
        coro_factory: Callable that returns an awaitable (called fresh each attempt).
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap in seconds.
        retryable_statuses: HTTP status codes that trigger a retry.
        retryable_exceptions: Exception types that trigger a retry.

    Returns:
        The successful httpx.Response.

    Raises:
        The last exception or httpx.HTTPStatusError after retries exhausted.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = await coro_factory()
            if response.status_code in retryable_statuses and attempt < max_retries:
                delay = _compute_delay(attempt, base_delay, max_delay)
                # Respect Retry-After header on 429
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                logger.warning(
                    "HTTP %d on attempt %d/%d, retrying in %.1fs",
                    response.status_code, attempt + 1, max_retries + 1, delay,
                )
                await asyncio.sleep(delay)
                continue
            return response
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = _compute_delay(attempt, base_delay, max_delay)
                logger.warning(
                    "%s on attempt %d/%d, retrying in %.1fs: %s",
                    type(exc).__name__, attempt + 1, max_retries + 1, delay, exc,
                )
                await asyncio.sleep(delay)
            else:
                raise
    # Should not reach here, but just in case
    raise last_exc  # type: ignore[misc]


def retry_sync(
    func: Callable[[], httpx.Response],
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
) -> httpx.Response:
    """Retry a sync HTTP call with exponential backoff and jitter.

    Args:
        func: Callable that performs the HTTP request.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap in seconds.
        retryable_exceptions: Exception types that trigger a retry.

    Returns:
        The successful httpx.Response.

    Raises:
        The last exception after retries exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except retryable_exceptions as exc:
            if attempt < max_retries:
                delay = _compute_delay(attempt, base_delay, max_delay)
                logger.warning(
                    "%s on attempt %d/%d, retrying in %.1fs: %s",
                    type(exc).__name__, attempt + 1, max_retries + 1, delay, exc,
                )
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("unreachable")  # pragma: no cover


def _compute_delay(attempt: int, base_delay: float, max_delay: float) -> float:
    """Compute delay with exponential backoff + jitter."""
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0, delay * 0.25)
    return delay + jitter
