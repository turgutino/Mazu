class MazuAPIError(Exception):
    """Base for all provider API errors, normalized across providers so callers
    (agent/loop.py, agent/autonomous.py, agent/council.py) can catch one type
    regardless of which provider is in use.
    """


class MazuRateLimitError(MazuAPIError):
    """Retryable: the provider rejected the request due to rate limiting."""


class MazuTransientError(MazuAPIError):
    """Retryable: network/timeout/connection issue, or a 5xx server error."""


class MazuAuthError(MazuAPIError):
    """Not retryable: missing, invalid, or expired API key."""


class MazuContextLengthError(MazuAPIError):
    """Not retryable without truncating the conversation: context window exceeded."""
