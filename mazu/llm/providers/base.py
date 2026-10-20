import threading
from abc import ABC, abstractmethod
from typing import Callable

from mazu.llm.types import AgentResponse


class Provider(ABC):
    """Every provider implements these methods against a canonical, Anthropic-shaped
    message/tool format (text/tool_use/tool_result content blocks). This is the seam:
    nothing outside mazu/llm/ needs to know which provider is actually in use.

    `api_key_env` names the environment variable this provider reads its key from, so
    generic code (config.ensure_api_key, memory extraction's provider-matching) can
    check/report the right thing without hardcoding a specific provider.

    `requires_api_key` defaults True for every existing (cloud) provider. A provider
    that talks to a server with no real auth (e.g. a local model server) sets this
    False so config.ensure_api_key() can skip demanding a key that doesn't exist for
    a good reason, without special-casing that provider by name anywhere.
    """

    api_key_env: str
    requires_api_key: bool = True

    def __init__(self):
        # `_client` is lazily built on first real use (mazu-web/mazu-windows run
        # every chat/run/explore/council session on its own thread, all sharing
        # this same Provider instance via client.py's module-level _PROVIDERS
        # dict), so the very first call from more than one thread at once could
        # previously race: both threads see `_client is None` and both construct
        # an SDK client concurrently, which corrupted process state badly enough
        # to crash Windows with a native heap-corruption exception (0xc0000374)
        # inside concurrent `ssl.create_default_context()` calls. The lock plus
        # re-check after acquiring it (`_get_or_create_client` below) is the
        # standard double-checked-locking fix -- cheap after the first call,
        # since later calls just read the already-set `_client` without ever
        # touching the lock.
        self._client = None
        self._client_lock = threading.Lock()

    def _get_or_create_client(self, factory: Callable[[], object]):
        """Thread-safe lazy-init helper shared by every provider's `_get_client()`.
        `factory` is called at most once per instance, even if many threads call
        this concurrently on their very first request.
        """
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = factory()
        return self._client

    # Whether this provider can act on the `effort` parameter below at all
    # ("low"/"medium"/"high"). False by default -- DeepSeek's and a local server's
    # APIs have no such knob, so `effort` is silently ignored for them rather than
    # erroring; only providers that override this to True (currently Anthropic,
    # Gemini, and OpenAIProvider specifically, not the OpenAICompatibleProvider
    # base OpenAI shares with DeepSeek/local) actually send it upstream.
    supports_reasoning_effort: bool = False

    def supports_reasoning_effort_for(self, model_name: str) -> bool:
        """Finer-grained than the class-level flag above: some providers whose API
        genuinely accepts `reasoning_effort` only do so for a subset of their own
        models (OpenAI: o-series/gpt-5 "reasoning models" accept it, gpt-4o and
        similar "chat models" reject it with a 400 -- see OpenAIProvider's
        override). Defaults to just returning the class-level flag, which is
        already model-independent for every provider that doesn't override this
        (Anthropic, Gemini: every model in the family accepts it; DeepSeek/local:
        no model does). `model_name` is the bare model id, no "provider:" prefix.
        """
        return self.supports_reasoning_effort

    def supports_vision_for(self, model_name: str) -> bool:
        """Best-effort, UI-only signal (mazu-web's attach-button warning) for
        whether `model_name` can actually see an attached image -- NOT consulted
        by run_turn/run_turn_stream themselves, which convert and send image
        blocks unconditionally regardless of this (see
        mazu/agent/context.py-adjacent image handling). Defaults to True
        (fail open): most current models across every provider here are
        multimodal, so assuming support is the safer default direction --
        a provider overrides this only where it has good reason to believe a
        real, common subset of its OWN models is text-only (see
        DeepSeekProvider's override for a concrete example).
        """
        return True

    @abstractmethod
    def run_turn(
        self, messages: list[dict], system: str, tools: list[dict], model: str,
        effort: str | None = None,
    ) -> AgentResponse: ...

    @abstractmethod
    def run_forced_tool(
        self, messages: list[dict], system: str, tool: dict, model: str
    ) -> tuple[dict, dict]:
        """Force the model to call exactly `tool`, returning (parsed input dict, usage
        dict). Used for structured extraction (see mazu/memory/extraction.py) where we
        need a guaranteed-shape result rather than an ordinary agentic turn. Usage is
        returned (not discarded) so callers can log real cost instead of this call's
        spend being invisible to `mazu usage`.
        """
        ...

    def run_turn_stream(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        model: str,
        on_delta: Callable[[str], None],
        effort: str | None = None,
    ) -> AgentResponse:
        """Default fallback for any provider that doesn't override this with real
        token-by-token streaming: runs the ordinary non-streaming call, then delivers
        the full text to `on_delta` in one shot once it's back. This guarantees every
        provider is callable through the streaming seam (a future provider that never
        gets a streaming override still works correctly, just without a live typing
        effect) instead of requiring every subclass to implement streaming to remain
        usable at all.
        """
        response = self.run_turn(messages, system, tools, model, effort=effort)
        text = "\n".join(b["text"] for b in response.content if b.get("type") == "text")
        if text:
            on_delta(text)
        return response

    def list_models(self) -> list[str]:
        """Returns the bare model ids this provider's own API currently offers to
        whichever key is configured (e.g. ["claude-opus-4-8", "claude-sonnet-5", ...]
        -- no "provider:" prefix, callers that need one add it themselves). This is a
        LIVE catalog call, not the static per-model capability table in
        mazu/llm/capabilities.py -- it exists so a model picker can be populated from
        what the account can actually use today instead of a hardcoded single default
        per provider going stale as new models ship.

        Raises the same MazuAuthError/MazuAPIError family as run_turn on a bad or
        missing key -- deliberately not swallowed into an empty list, since an empty
        list would look identical to "this account genuinely has zero models" instead
        of "your key is wrong."
        """
        raise NotImplementedError(f"{type(self).__name__} does not support live model listing")
