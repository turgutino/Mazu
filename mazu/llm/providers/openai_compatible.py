import json
import os

from mazu.llm.error_mapping import classify_sdk_error
from mazu.llm.errors import MazuAPIError, MazuAuthError
from mazu.llm.providers.base import Provider
from mazu.llm.types import AgentResponse


def _to_openai_tools(tools: list[dict]) -> list[dict] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """Translate the canonical Anthropic-shaped message list into OpenAI's chat format.
    Anthropic tool_result blocks (embedded in a user turn) become their own `tool`-role
    messages; Anthropic tool_use blocks (embedded in an assistant turn) become `tool_calls`.
    """
    openai_messages: list[dict] = [{"role": "system", "content": system}]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text_parts = [b["text"] for b in content if b.get("type") == "text"]
            tool_calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                }
                for b in content
                if b.get("type") == "tool_use"
            ]
            entry: dict = {"role": "assistant", "content": "\n".join(text_parts) or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            openai_messages.append(entry)
        else:  # user turn: may contain tool_result blocks
            for block in content:
                if block.get("type") == "tool_result":
                    openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": str(block.get("content", "")),
                        }
                    )
                elif block.get("type") == "text":
                    openai_messages.append({"role": "user", "content": block["text"]})
    return openai_messages


def _parse_tool_arguments(raw: str | None) -> dict:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        raise MazuAPIError(f"Model returned malformed tool-call JSON: {e}") from e


class OpenAICompatibleProvider(Provider):
    """Base for any provider exposing an OpenAI-compatible chat completions API —
    OpenAI itself, DeepSeek, and others that follow the same wire contract. Subclasses
    just fix a base_url and which env var holds the API key; the request/response
    handling and message/tool conversion are shared.
    """

    def __init__(self, base_url: str | None, api_key_env: str, provider_label: str):
        self._client = None
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.provider_label = provider_label

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                # MazuAuthError isn't a perfect semantic fit for "missing dependency," but
                # it's a "can't proceed with this provider, not retryable" condition like
                # the others below, and reusing it means callers only need to catch one
                # exception type to handle all "provider isn't usable" cases gracefully.
                raise MazuAuthError(
                    "The openai package isn't installed. Run `pip install mazu[openai]` "
                    f"(or `pip install openai`) to use {self.provider_label} models."
                ) from e
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise MazuAuthError(
                    f"{self.api_key_env} is not set. Set it to use {self.provider_label} models."
                )
            if not api_key.isascii():
                # A real API key is always plain ASCII. Non-ASCII here means the env var
                # holds something else entirely (a pasted placeholder, a typo, leftover
                # example text) -- left unchecked, this fails much later and much more
                # confusingly, deep inside httpx's header encoding with a raw
                # UnicodeEncodeError instead of a clear message about the actual problem.
                raise MazuAuthError(
                    f"{self.api_key_env} contains non-ASCII characters, so it can't be a "
                    "real API key. It looks like it was set to a placeholder or example "
                    f"value by mistake. Set {self.api_key_env} to your actual key and try again."
                )
            self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    def run_turn(
        self, messages: list[dict], system: str, tools: list[dict], model: str
    ) -> AgentResponse:
        import openai

        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=_to_openai_messages(system, messages),
                tools=_to_openai_tools(tools),
            )
        except openai.OpenAIError as e:
            raise classify_sdk_error(openai, e) from e

        if not response.choices:
            raise MazuAPIError(f"{self.provider_label} response contained no choices")
        choice = response.choices[0]

        content_blocks = []
        if choice.message.content:
            content_blocks.append({"type": "text", "text": choice.message.content})
        for tool_call in choice.message.tool_calls or []:
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": _parse_tool_arguments(tool_call.function.arguments),
                }
            )

        stop_reason = "tool_use" if choice.message.tool_calls else "end_turn"
        usage = response.usage.model_dump() if response.usage else {}
        return AgentResponse(stop_reason=stop_reason, content=content_blocks, usage=usage)

    def run_forced_tool(
        self, messages: list[dict], system: str, tool: dict, model: str
    ) -> dict:
        import openai

        client = self._get_client()
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        try:
            response = client.chat.completions.create(
                model=model,
                messages=_to_openai_messages(system, messages),
                tools=[openai_tool],
                tool_choice={"type": "function", "function": {"name": tool["name"]}},
            )
        except openai.OpenAIError as e:
            raise classify_sdk_error(openai, e) from e

        if not response.choices:
            raise MazuAPIError(f"{self.provider_label} response contained no choices")
        for tool_call in response.choices[0].message.tool_calls or []:
            if tool_call.function.name == tool["name"]:
                return _parse_tool_arguments(tool_call.function.arguments)
        return {}
