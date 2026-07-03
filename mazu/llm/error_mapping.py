from mazu.llm.errors import (
    MazuAPIError,
    MazuAuthError,
    MazuContextLengthError,
    MazuRateLimitError,
    MazuTransientError,
)

# Anthropic's and OpenAI's Python SDKs happen to expose near-identical exception
# class names (RateLimitError, AuthenticationError, APIConnectionError,
# APITimeoutError, APIStatusError), so one classifier works for both -- pass in
# the already-imported SDK module (`anthropic` or `openai`).
_CONTEXT_LENGTH_HINTS = (
    "context length",
    "context_length",
    "maximum context",
    "too long",
    "too many tokens",
)


def classify_sdk_error(sdk, error: Exception) -> MazuAPIError:
    if isinstance(error, sdk.RateLimitError):
        return MazuRateLimitError(str(error))
    if isinstance(error, sdk.AuthenticationError):
        return MazuAuthError(str(error))
    if isinstance(error, (sdk.APIConnectionError, sdk.APITimeoutError)):
        return MazuTransientError(str(error))
    if isinstance(error, sdk.APIStatusError):
        message = str(error).lower()
        if any(hint in message for hint in _CONTEXT_LENGTH_HINTS):
            return MazuContextLengthError(str(error))
        if error.status_code >= 500:
            return MazuTransientError(str(error))
        return MazuAPIError(str(error))
    return MazuAPIError(str(error))
