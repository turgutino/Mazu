import json
import os
from typing import Callable

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


def _image_block_to_openai_part(block: dict) -> dict:
    """Anthropic's canonical `{"type": "image", "source": {"type": "base64",
    "media_type": ..., "data": ...}}` -> OpenAI's `image_url` content part, which
    accepts a base64 image inline as a `data:` URI (documented OpenAI vision
    input format -- no separate upload step needed for this shape).
    """
    source = block.get("source", {})
    media_type = source.get("media_type", "image/png")
    data = source.get("data", "")
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}


def _to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """Translate the canonical Anthropic-shaped message list into OpenAI's chat format.
    Anthropic tool_result blocks (embedded in a user turn) become their own `tool`-role
    messages; Anthropic tool_use blocks (embedded in an assistant turn) become `tool_calls`;
    image blocks become `image_url` parts alongside any text parts in the same user turn.
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
        else:
            # user turn: may contain tool_result blocks (each becomes its own
            # `tool`-role message, in source order relative to the surrounding
            # text/image parts) and text/image blocks, which are collected into
            # ONE user message so an image and its caption stay in the same
            # turn -- OpenAI's vision input requires that shape; splitting them
            # into separate messages the way the old text-only code did would
            # silently break the image-to-caption association.
            parts: list[dict] = []
            for block in content:
                block_type = block.get("type")
                if block_type == "tool_result":
                    openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": str(block.get("content", "")),
                        }
                    )
                elif block_type == "text":
                    parts.append({"type": "text", "text": block["text"]})
                elif block_type == "image":
                    parts.append(_image_block_to_openai_part(block))
            if parts:
                # A single plain-text part keeps the old bare-string shape --
                # some OpenAI-compatible servers are stricter about accepting
                # array content for pure-text messages than the real OpenAI API.
                if len(parts) == 1 and parts[0]["type"] == "text":
                    openai_messages.append({"role": "user", "content": parts[0]["text"]})
                else:
                    openai_messages.append({"role": "user", "content": parts})
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

    # A class attribute (not a module constant) so LocalProvider can override it
    # with a longer value instead of sharing one number -- a cloud request to
    # deepseek/etc and a request to a CPU-bound local model on the user's own
    # machine have very different realistic response times. Exists because the SDK's own
    # default timeout (several minutes) previously left a stalled request hanging
    # with no recovery and no feedback: caught live via a mazu-web chat session
    # (routed through this provider for deepseek: models) stuck unresponsive for
    # 8+ minutes. 120s is generous for a single request/streamed-chunk gap on a
    # cloud API while still failing fast enough that MazuAPIError's existing
    # retry/error-surfacing path (mazu/llm/retry.py) gets a chance to run at all.
    REQUEST_TIMEOUT_SECONDS = 120.0

    def __init__(
        self,
        base_url: str | None,
        api_key_env: str,
        provider_label: str,
        requires_api_key: bool = True,
    ):
        super().__init__()
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.provider_label = provider_label
        self.requires_api_key = requires_api_key

    def _get_client(self):
        def factory():
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
            if self.requires_api_key:
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
            else:
                # A provider that opts out of requiring a key (e.g. a local model
                # server with no real auth) still needs *some* string handed to the
                # OpenAI SDK's client constructor -- it always sends an Authorization
                # header. "not-needed" is never surfaced to the user; if they DO set
                # the env var (e.g. a local proxy that actually checks a token),
                # their real value is used instead.
                api_key = os.environ.get(self.api_key_env) or "not-needed"
            return OpenAI(api_key=api_key, base_url=self.base_url, timeout=self.REQUEST_TIMEOUT_SECONDS)

        return self._get_or_create_client(factory)

    def _effort_kwargs(self, effort: str | None) -> dict:
        """Only OpenAIProvider overrides `supports_reasoning_effort` to True --
        DeepSeek's and a local server's chat-completions endpoints have no such
        parameter, so `effort` is silently dropped for them rather than sent
        upstream (where an OpenAI-compatible server might reject an unrecognized
        field, or a lenient one would just ignore it -- not worth relying on
        either behavior when simply not sending it is unambiguous).
        """
        if effort and self.supports_reasoning_effort:
            return {"reasoning_effort": effort}
        return {}

    def run_turn(
        self, messages: list[dict], system: str, tools: list[dict], model: str,
        effort: str | None = None,
    ) -> AgentResponse:
        # _get_client() must run first: it's the one that turns a missing `openai`
        # package into a clean MazuAuthError. Importing it unconditionally here would
        # raise a raw ModuleNotFoundError before that guard ever runs.
        client = self._get_client()
        import openai

        try:
            response = client.chat.completions.create(
                model=model,
                messages=_to_openai_messages(system, messages),
                tools=_to_openai_tools(tools),
                **self._effort_kwargs(effort),
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

    def run_turn_stream(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        model: str,
        on_delta: Callable[[str], None],
        effort: str | None = None,
    ) -> AgentResponse:
        client = self._get_client()
        import openai

        text_parts: list[str] = []
        # Tool-call arguments arrive as fragmented JSON string deltas, keyed by their
        # position in the response (`index`), not by id -- the id/name may only appear
        # on the first delta for that index, with further deltas carrying just an
        # `arguments` fragment to append. Must accumulate by index before parsing.
        tool_calls: dict[int, dict] = {}
        usage: dict = {}
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=_to_openai_messages(system, messages),
                tools=_to_openai_tools(tools),
                stream=True,
                stream_options={"include_usage": True},
                **self._effort_kwargs(effort),
            )
            for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage.model_dump()
                if not chunk.choices:
                    continue  # the final usage-only chunk has an empty choices list
                delta = chunk.choices[0].delta
                if delta.content:
                    text_parts.append(delta.content)
                    on_delta(delta.content)
                for tc_delta in delta.tool_calls or []:
                    slot = tool_calls.setdefault(
                        tc_delta.index, {"id": None, "name": None, "arguments": ""}
                    )
                    if tc_delta.id:
                        slot["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        slot["name"] = tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        slot["arguments"] += tc_delta.function.arguments
        except openai.OpenAIError as e:
            raise classify_sdk_error(openai, e) from e

        content_blocks = []
        text = "".join(text_parts)
        if text:
            content_blocks.append({"type": "text", "text": text})
        for index in sorted(tool_calls):
            tc = tool_calls[index]
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": _parse_tool_arguments(tc["arguments"]),
                }
            )

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return AgentResponse(stop_reason=stop_reason, content=content_blocks, usage=usage)

    def run_forced_tool(
        self, messages: list[dict], system: str, tool: dict, model: str
    ) -> tuple[dict, dict]:
        client = self._get_client()
        import openai

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
        usage = response.usage.model_dump() if response.usage else {}
        for tool_call in response.choices[0].message.tool_calls or []:
            if tool_call.function.name == tool["name"]:
                return _parse_tool_arguments(tool_call.function.arguments), usage
        return {}, usage

    def _is_chat_model(self, model_id: str) -> bool:
        """Whether a model id from list_models()'s raw catalog should be offered as
        a chat model. True for everything by default -- DeepSeek's and a local
        server's own catalogs are already small and chat-only, so no filtering is
        needed there. OpenAIProvider overrides this: OpenAI's full /v1/models
        response also mixes in embeddings, TTS, image, moderation, and legacy
        completion-only models alongside genuine chat models.
        """
        return True

    def list_models(self) -> list[str]:
        client = self._get_client()
        import openai

        try:
            response = client.models.list()
        except openai.OpenAIError as e:
            raise classify_sdk_error(openai, e) from e
        return sorted(m.id for m in response.data if self._is_chat_model(m.id))
