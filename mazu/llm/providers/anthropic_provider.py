import os
from typing import Callable

from mazu.llm.capabilities import DEFAULT_ANTHROPIC_MAX_TOKENS, MAX_OUTPUT_TOKENS
from mazu.llm.error_mapping import classify_sdk_error
from mazu.llm.errors import MazuAuthError
from mazu.llm.providers.base import Provider
from mazu.llm.types import AgentResponse

# Structured single-field extraction (run_forced_tool, used only by
# mazu/memory/extraction.py) never needs a long free-form output -- kept as its
# own small, fixed constant, independent of the per-model table below, which
# exists specifically for ordinary agentic turns (run_turn/run_turn_stream) that
# can legitimately need a long response (a big diff, a long explanation).
FORCED_TOOL_MAX_TOKENS = 1024

# The Anthropic SDK's own default (10 minutes) is long enough that a genuinely
# stalled connection -- not a slow-but-progressing generation, which streaming
# chunks reset this against -- leaves a chat/run/council turn hung with zero
# feedback and no way to recover short of restarting the whole mazu-web process.
# Caught live: a mazu-web chat session sat unresponsive for 8+ minutes with no
# self-recovery. 120s is generous for a single request/chunk gap while still
# failing fast enough that MazuAPIError's existing retry/error-surfacing path
# (see mazu/llm/retry.py) actually gets a chance to run instead of the call never
# returning at all.
_REQUEST_TIMEOUT_SECONDS = 120.0


def _max_output_tokens(model: str) -> int:
    """Anthropic's `max_tokens` request field is required and hard-caps a single
    turn's output -- unlike OpenAI-compatible/Gemini, which need no equivalent
    parameter here at all. A flat constant either wastes a capable model's real
    ceiling or (as the old MAX_TOKENS = 4096 did) truncates one that supports far
    more, which would silently look like a quality failure in a fair cross-model
    benchmark rather than the artificial limit it actually is. See
    mazu/llm/capabilities.py's MAX_OUTPUT_TOKENS for the model table and its
    "best-effort, will go stale" caveat.
    """
    return MAX_OUTPUT_TOKENS.get(f"anthropic:{model}", DEFAULT_ANTHROPIC_MAX_TOKENS)


def _with_cache_control(message: dict) -> dict:
    """Returns a NEW message dict with a cache_control breakpoint on its last content
    block -- never mutates the given message or any nested content-block dict, since
    the caller's `messages` list is the same object reused verbatim elsewhere
    (checkpoint snapshots, context compaction, the OpenAI-compatible message
    converter). This is the standard Anthropic multi-turn caching pattern: moving the
    breakpoint to the last message on every request means each new request's cache
    read covers everything through the previous turn, and only the newest increment
    is priced at full, uncached rate.

    `content` here is one of two shapes seen in this codebase: a plain string (the
    first user turn, `{"role": "user", "content": task}`), or a list of content-block
    dicts (an assistant turn's `response.content`, or a tool-result round's
    `tool_results` list). Both need converting into "a list of blocks whose last
    element carries cache_control" -- a bare string can't carry the key itself.
    """
    content = message["content"]
    if isinstance(content, str):
        new_content = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
    else:
        new_content = [*content[:-1], {**content[-1], "cache_control": {"type": "ephemeral"}}]
    return {**message, "content": new_content}


def _effort_kwargs(effort: str | None) -> dict:
    """Mazu's own three-level vocabulary ("low"/"medium"/"high") mapped onto
    current-generation Claude models' `thinking`/`output_config.effort` request
    shape (verified against the installed anthropic SDK and the claude-api skill's
    Thinking & Effort reference -- `budget_tokens` is removed/rejected on these
    models, `output_config.effort` is the current GA replacement). Returns {} when
    `effort` is unset, leaving the model's own default thinking behavior untouched.
    """
    if not effort:
        return {}
    return {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}


def _with_tool_cache_control(tools: list[dict]) -> list[dict]:
    """Returns a NEW tools list with a cache_control breakpoint on the last tool
    definition -- caches the entire (large, static-per-run) tool schema block.
    Never mutates the caller's `tools` list or its dicts.
    """
    if not tools:
        return tools
    return [*tools[:-1], {**tools[-1], "cache_control": {"type": "ephemeral"}}]


class AnthropicProvider(Provider):
    supports_reasoning_effort = True

    def __init__(self):
        self._client = None
        self.api_key_env = "ANTHROPIC_API_KEY"

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic

            api_key = os.environ.get(self.api_key_env)
            if api_key and not api_key.isascii():
                # A real API key is always plain ASCII. Non-ASCII here means the env var
                # holds something else (a pasted placeholder, a typo, leftover example
                # text) -- left unchecked, this fails much later and far more confusingly,
                # deep inside the HTTP client's header encoding.
                raise MazuAuthError(
                    f"{self.api_key_env} contains non-ASCII characters, so it can't be a "
                    "real API key. It looks like it was set to a placeholder or example "
                    f"value by mistake. Set {self.api_key_env} to your actual key and try again."
                )
            self._client = Anthropic(api_key=api_key, timeout=_REQUEST_TIMEOUT_SECONDS)
        return self._client

    def run_turn(
        self, messages: list[dict], system: str, tools: list[dict], model: str,
        effort: str | None = None,
    ) -> AgentResponse:
        import anthropic

        client = self._get_client()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=_max_output_tokens(model),
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[*messages[:-1], _with_cache_control(messages[-1])] if messages else messages,
                tools=_with_tool_cache_control(tools),
                **_effort_kwargs(effort),
            )
        except anthropic.AnthropicError as e:
            raise classify_sdk_error(anthropic, e) from e
        return AgentResponse(
            stop_reason=response.stop_reason,
            content=[block.model_dump() for block in response.content],
            usage=response.usage.model_dump(),
        )

    def run_turn_stream(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        model: str,
        on_delta: Callable[[str], None],
        effort: str | None = None,
    ) -> AgentResponse:
        import anthropic

        client = self._get_client()
        try:
            with client.messages.stream(
                model=model,
                max_tokens=_max_output_tokens(model),
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[*messages[:-1], _with_cache_control(messages[-1])] if messages else messages,
                tools=_with_tool_cache_control(tools),
                **_effort_kwargs(effort),
            ) as stream:
                for text in stream.text_stream:
                    on_delta(text)
                # get_final_message() returns the exact same shape as messages.create()'s
                # return value, so it goes through identical parsing to the non-streaming
                # path below -- streaming only changes *when* text is delivered, not how
                # the final content blocks (including tool_use) are built.
                final = stream.get_final_message()
        except anthropic.AnthropicError as e:
            raise classify_sdk_error(anthropic, e) from e
        return AgentResponse(
            stop_reason=final.stop_reason,
            content=[block.model_dump() for block in final.content],
            usage=final.usage.model_dump(),
        )

    def run_forced_tool(
        self, messages: list[dict], system: str, tool: dict, model: str
    ) -> tuple[dict, dict]:
        import anthropic

        client = self._get_client()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=FORCED_TOOL_MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=messages,
                tools=[tool],
                tool_choice={"type": "tool", "name": tool["name"]},
            )
        except anthropic.AnthropicError as e:
            raise classify_sdk_error(anthropic, e) from e
        usage = response.usage.model_dump()
        for block in response.content:
            if block.type == "tool_use" and block.name == tool["name"]:
                return block.input, usage
        return {}, usage

    def list_models(self) -> list[str]:
        import anthropic

        client = self._get_client()
        try:
            # The SDK's returned page auto-fetches subsequent pages as it's
            # iterated, so this collects the full catalog, not just the first page.
            models = [m.id for m in client.models.list()]
        except anthropic.AnthropicError as e:
            raise classify_sdk_error(anthropic, e) from e
        return sorted(models, reverse=True)
