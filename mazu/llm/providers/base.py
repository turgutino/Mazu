from abc import ABC, abstractmethod

from mazu.llm.types import AgentResponse


class Provider(ABC):
    """Every provider implements these methods against a canonical, Anthropic-shaped
    message/tool format (text/tool_use/tool_result content blocks). This is the seam:
    nothing outside mazu/llm/ needs to know which provider is actually in use.

    `api_key_env` names the environment variable this provider reads its key from, so
    generic code (config.ensure_api_key, memory extraction's provider-matching) can
    check/report the right thing without hardcoding a specific provider.
    """

    api_key_env: str

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
