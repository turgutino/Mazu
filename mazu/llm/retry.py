import time
from typing import Callable, TypeVar

from mazu.llm.errors import MazuRateLimitError, MazuTransientError

T = TypeVar("T")

RETRYABLE = (MazuRateLimitError, MazuTransientError)


def with_retry(fn: Callable[[], T], max_attempts: int = 3, base_delay: float = 1.0) -> T:
    """Retries `fn` with exponential backoff (1s, 2s, 4s, ...) only for the two
    error types that are actually worth retrying (rate limits, transient network/
    server issues). Anything else (auth, context-length, unknown) propagates
    immediately — retrying those would just waste time and API calls.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except RETRYABLE as e:
            last_exc = e
            if attempt < max_attempts - 1:
                time.sleep(base_delay * (2**attempt))
    assert last_exc is not None
    raise last_exc
