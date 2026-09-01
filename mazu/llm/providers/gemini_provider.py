import os

from mazu.llm.error_mapping import _CONTEXT_LENGTH_HINTS
from mazu.llm.errors import (
    MazuAPIError,
    MazuAuthError,
    MazuContextLengthError,
    MazuRateLimitError,
    MazuTransientError,
)
from mazu.llm.providers.base import Provider
from mazu.llm.types import AgentResponse


def _to_gemini_tools(tools: list[dict]):
    if not tools:
        return None
    from google.genai import types

    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters_json_schema=t["input_schema"],
                )
                for t in tools
            ]
        )
    ]


def _to_gemini_contents(messages: list[dict]):
    """Translates the canonical Anthropic-shaped message list into Gemini's Content/
    Part format. Two structural differences from Anthropic/OpenAI that make this
    conversion non-trivial:
    1. Gemini uses role "model" where Anthropic/OpenAI use "assistant".
    2. Gemini's function_response Part needs the function's *name*, but Anthropic's
       tool_result block only carries the tool_use_id -- the name has to be looked
       up from the tool_use block that originally used that id, tracked here as we
       walk forward through the conversation.
    """
    from google.genai import types

    contents = []
    tool_name_by_id: dict[str, str] = {}

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(types.Content(role=gemini_role, parts=[types.Part.from_text(text=content)]))
            continue

        if role == "assistant":
            parts = []
            for block in content:
                if block.get("type") == "text":
                    parts.append(types.Part.from_text(text=block["text"]))
                elif block.get("type") == "tool_use":
                    tool_name_by_id[block["id"]] = block["name"]
                    parts.append(types.Part.from_function_call(name=block["name"], args=block["input"]))
            if parts:
                contents.append(types.Content(role="model", parts=parts))
        else:  # user turn: plain text, image blocks, or tool_result blocks from a prior tool round
            parts = []
            for block in content:
                if block.get("type") == "tool_result":
                    name = tool_name_by_id.get(block["tool_use_id"], "unknown_function")
                    parts.append(
                        types.Part.from_function_response(
                            name=name, response={"result": str(block.get("content", ""))}
                        )
                    )
                elif block.get("type") == "text":
                    parts.append(types.Part.from_text(text=block["text"]))
                elif block.get("type") == "image":
                    # Anthropic's canonical {"source": {"media_type", "data": <base64>}}
                    # -- Part.from_bytes wants raw bytes, not the base64 string itself.
                    import base64

                    source = block.get("source", {})
                    parts.append(
                        types.Part.from_bytes(
                            data=base64.b64decode(source.get("data", "")),
                            mime_type=source.get("media_type", "image/png"),
                        )
                    )
            if parts:
                contents.append(types.Content(role="user", parts=parts))

    return contents


def _parse_gemini_response(response) -> AgentResponse:
    content_blocks = []
    candidates = response.candidates or []
    parts = candidates[0].content.parts if candidates and candidates[0].content else []
    for part in parts or []:
        if part.text:
            content_blocks.append({"type": "text", "text": part.text})
        elif part.function_call:
            fc = part.function_call
            content_blocks.append(
                {
                    # Gemini doesn't assign its own call id the way Anthropic/OpenAI do;
                    # synthesize one so downstream tool_result matching (which keys off
                    # this id) still works. Name+index is unique within a single turn,
                    # which is all that's needed since ids never need to survive past it.
                    "type": "tool_use",
                    "id": fc.id or f"{fc.name}_{len(content_blocks)}",
                    "name": fc.name,
                    "input": dict(fc.args or {}),
                }
            )

    stop_reason = "tool_use" if any(b["type"] == "tool_use" for b in content_blocks) else "end_turn"

    usage_obj = response.usage_metadata
    usage = {}
    if usage_obj is not None:
        usage = {
            "input_tokens": usage_obj.prompt_token_count or 0,
            "output_tokens": usage_obj.candidates_token_count or 0,
            # Only ever surfaces what Gemini itself reports (implicit, automatic
            # prompt-prefix caching -- Mazu requests no explicit CachedContent).
            # getattr guards against an older google-genai SDK version whose
            # UsageMetadata predates this field; absent/None both mean "no cache
            # data available", same as every other provider's cache fields --
            # never estimated or assumed non-zero.
            "cached_content_token_count": getattr(usage_obj, "cached_content_token_count", None) or 0,
        }

    return AgentResponse(stop_reason=stop_reason, content=content_blocks, usage=usage)


def _thinking_config(effort: str | None):
    """Mazu's own three-level vocabulary ("low"/"medium"/"high") mapped onto the
    installed google-genai SDK's ThinkingConfig.thinking_level enum (verified via
    `from google.genai import types; types.ThinkingLevel` -- MINIMAL/LOW/MEDIUM/
    HIGH), Gemini's direct equivalent of Anthropic's output_config.effort and
    OpenAI's reasoning_effort. Returns None when `effort` is unset, leaving
    thinking_config entirely off the request -- same as omitting it.
    """
    if not effort:
        return None
    from google.genai import types

    return types.ThinkingConfig(thinking_level=effort.upper())


def _classify_gemini_error(error: Exception) -> MazuAPIError:
    from google.genai import errors as genai_errors

    if not isinstance(error, genai_errors.APIError):
        return MazuAPIError(str(error))

    message = str(error)
    code = getattr(error, "code", None)
    if code in (401, 403):
        return MazuAuthError(message)
    if code == 429:
        return MazuRateLimitError(message)
    if code is not None and code >= 500:
        return MazuTransientError(message)
    if any(hint in message.lower() for hint in _CONTEXT_LENGTH_HINTS):
        return MazuContextLengthError(message)
    return MazuAPIError(message)


class GeminiProvider(Provider):
    """Deliberately does NOT override run_turn_stream -- unlike Anthropic's
    stream()/get_final_message() pattern (which reuses the exact same response
    parsing as the non-streaming path, verified safe) or OpenAI's well-documented
    incremental function-call-argument deltas, Gemini's chunk-level behavior for
    function calls during streaming wasn't something that could be verified
    correct from static SDK inspection alone. Falls back to Provider's default
    (call run_turn, deliver the full text to on_delta once) -- fully correct,
    just without a live typing effect. Revisit with real streaming once that
    chunk behavior can be verified against the live API.
    """

    supports_reasoning_effort = True

    def __init__(self):
        self._client = None
        self.api_key_env = "GEMINI_API_KEY"

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as e:
                raise MazuAuthError(
                    "The google-genai package isn't installed. Run "
                    "`pip install mazu[gemini]` (or `pip install google-genai`) "
                    "to use gemini: models."
                ) from e
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise MazuAuthError(f"{self.api_key_env} is not set. Set it to use gemini: models.")
            if not api_key.isascii():
                raise MazuAuthError(
                    f"{self.api_key_env} contains non-ASCII characters, so it can't be a "
                    "real API key. It looks like it was set to a placeholder or example "
                    f"value by mistake. Set {self.api_key_env} to your actual key and try again."
                )
            # timeout is in MILLISECONDS for this SDK (google.genai.types.HttpOptions'
            # own docstring), unlike anthropic/openai's plain-seconds `timeout=` kwarg
            # -- see anthropic_provider.py's _REQUEST_TIMEOUT_SECONDS for why a finite
            # timeout matters here at all (a stalled request previously hung forever
            # with the SDK's own default, caught live via an 8+ minute unresponsive
            # mazu-web chat session).
            self._client = genai.Client(
                api_key=api_key, http_options=types.HttpOptions(timeout=120_000)
            )
        return self._client

    def run_turn(
        self, messages: list[dict], system: str, tools: list[dict], model: str,
        effort: str | None = None,
    ) -> AgentResponse:
        client = self._get_client()
        from google.genai import types

        try:
            response = client.models.generate_content(
                model=model,
                contents=_to_gemini_contents(messages),
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    tools=_to_gemini_tools(tools),
                    thinking_config=_thinking_config(effort),
                ),
            )
        except Exception as e:
            raise _classify_gemini_error(e) from e
        return _parse_gemini_response(response)

    def run_forced_tool(
        self, messages: list[dict], system: str, tool: dict, model: str
    ) -> tuple[dict, dict]:
        client = self._get_client()
        from google.genai import types

        try:
            response = client.models.generate_content(
                model=model,
                contents=_to_gemini_contents(messages),
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    tools=_to_gemini_tools([tool]),
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="ANY", allowed_function_names=[tool["name"]]
                        )
                    ),
                ),
            )
        except Exception as e:
            raise _classify_gemini_error(e) from e

        parsed = _parse_gemini_response(response)
        for block in parsed.content:
            if block["type"] == "tool_use" and block["name"] == tool["name"]:
                return block["input"], parsed.usage
        return {}, parsed.usage

    def list_models(self) -> list[str]:
        client = self._get_client()
        try:
            # The returned Pager auto-fetches subsequent pages as it's iterated, so
            # this collects the full catalog, not just the first page.
            catalog = list(client.models.list())
        except Exception as e:
            raise _classify_gemini_error(e) from e

        model_ids = []
        for m in catalog:
            # Gemini's catalog also lists embedding/imagen/veo/etc models that can't
            # be used with generate_content -- supported_actions is how the API
            # itself marks which ones actually accept a chat-style call.
            if "generateContent" not in (m.supported_actions or []):
                continue
            name = m.name or ""
            model_ids.append(name.removeprefix("models/"))
        return sorted(model_ids)
