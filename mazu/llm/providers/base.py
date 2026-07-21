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

    @abstractmethod
    def run_turn(
        self, messages: list[dict], system: str, tools: list[dict], model: str
    ) -> AgentResponse: ...

    @abstractmethod
    def run_forced_tool(
        self, messages: list[dict], system: str, tool: dict, model: str
    ) -> dict:
        """Force the model to call exactly `tool`, returning its parsed input dict.
        Used for structured extraction (see mazu/memory/extraction.py) where we need a
        guaranteed-shape result rather than an ordinary agentic turn.
        """
        ...

    def run_turn_stream(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        model: str,
        on_delta: Callable[[str], None],
    ) -> AgentResponse:
        """Default fallback for any provider that doesn't override this with real
        token-by-token streaming: runs the ordinary non-streaming call, then delivers
        the full text to `on_delta` in one shot once it's back. This guarantees every
        provider is callable through the streaming seam (a future provider that never
        gets a streaming override still works correctly, just without a live typing
        effect) instead of requiring every subclass to implement streaming to remain
        usable at all.
        """
        response = self.run_turn(messages, system, tools, model)
        text = "\n".join(b["text"] for b in response.content if b.get("type") == "text")
        if text:
            on_delta(text)
        return response
