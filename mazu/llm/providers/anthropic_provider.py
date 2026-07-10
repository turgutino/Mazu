import os
from typing import Callable

from mazu.llm.error_mapping import classify_sdk_error
from mazu.llm.errors import MazuAuthError
from mazu.llm.providers.base import Provider
from mazu.llm.types import AgentResponse

MAX_TOKENS = 4096


class AnthropicProvider(Provider):
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
            self._client = Anthropic()
        return self._client

    def run_turn(
        self, messages: list[dict], system: str, tools: list[dict], model: str
    ) -> AgentResponse:
        import anthropic

        client = self._get_client()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=messages,
                tools=tools,
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
    ) -> AgentResponse:
        import anthropic

        client = self._get_client()
        try:
            with client.messages.stream(
                model=model,
                max_tokens=MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=messages,
                tools=tools,
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
    ) -> dict:
        import anthropic

        client = self._get_client()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=system,
                messages=messages,
                tools=[tool],
                tool_choice={"type": "tool", "name": tool["name"]},
            )
        except anthropic.AnthropicError as e:
            raise classify_sdk_error(anthropic, e) from e
        for block in response.content:
            if block.type == "tool_use" and block.name == tool["name"]:
                return block.input
        return {}
