"""Tests for the Gemini provider. Uses the REAL google-genai SDK types (not fakes)
for constructing test data wherever practical -- Gemini's Content/Part/FunctionCall
shapes are different enough from Anthropic/OpenAI's (role name is "model" not
"assistant", tool_result needs a function *name* looked up from the id, error
classes are structured differently) that hand-rolled fakes would risk silently
drifting from what the real SDK actually returns.
"""

import sys
from unittest.mock import MagicMock

import pytest
from google.genai import errors as genai_errors
from google.genai import types

from mazu.llm.errors import (
    MazuAuthError,
    MazuContextLengthError,
    MazuRateLimitError,
    MazuTransientError,
)
from mazu.llm.providers.gemini_provider import (
    GeminiProvider,
    _classify_gemini_error,
    _parse_gemini_response,
    _to_gemini_contents,
    _to_gemini_tools,
)

# ---------------------------------------------------------------------------
# _to_gemini_contents
# ---------------------------------------------------------------------------


def test_plain_user_text_message():
    contents = _to_gemini_contents([{"role": "user", "content": "hello"}])
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "hello"


def test_assistant_role_becomes_model():
    contents = _to_gemini_contents(
        [{"role": "assistant", "content": [{"type": "text", "text": "hi there"}]}]
    )
    assert contents[0].role == "model"
    assert contents[0].parts[0].text == "hi there"


def test_tool_use_converts_to_function_call():
    contents = _to_gemini_contents(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}}
                ],
            }
        ]
    )
    assert contents[0].role == "model"
    fc = contents[0].parts[0].function_call
    assert fc.name == "read_file"
    assert fc.args == {"path": "a.py"}


def test_tool_result_looks_up_function_name_from_prior_tool_use():
    """The core correctness risk in this conversion: Anthropic's tool_result block
    only carries a tool_use_id, but Gemini's function_response needs the function's
    *name* -- it must be looked up from the matching tool_use block seen earlier.
    """
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_abc", "name": "read_file", "input": {"path": "a.py"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_abc", "content": "file contents"}],
        },
    ]
    contents = _to_gemini_contents(messages)
    assert contents[1].role == "user"
    fr = contents[1].parts[0].function_response
    assert fr.name == "read_file"
    assert fr.response == {"result": "file contents"}


def test_multiple_tool_calls_track_names_independently():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_a", "name": "read_file", "input": {"path": "a.py"}},
                {"type": "tool_use", "id": "call_b", "name": "list_dir", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_b", "content": "dir listing"},
                {"type": "tool_result", "tool_use_id": "call_a", "content": "file contents"},
            ],
        },
    ]
    contents = _to_gemini_contents(messages)
    responses = contents[1].parts
    assert responses[0].function_response.name == "list_dir"
    assert responses[1].function_response.name == "read_file"


# ---------------------------------------------------------------------------
# _to_gemini_tools
# ---------------------------------------------------------------------------


def test_empty_tools_returns_none():
    assert _to_gemini_tools([]) is None


def test_tool_schema_conversion():
    tools = _to_gemini_tools(
        [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            }
        ]
    )
    assert len(tools) == 1
    decl = tools[0].function_declarations[0]
    assert decl.name == "read_file"
    assert decl.description == "Read a file"
    assert decl.parameters_json_schema == {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }


# ---------------------------------------------------------------------------
# _parse_gemini_response
# ---------------------------------------------------------------------------


def _make_response(parts, usage=None):
    content = types.Content(role="model", parts=parts)
    candidate = types.Candidate(content=content)
    return types.GenerateContentResponse(candidates=[candidate], usage_metadata=usage)


def test_parse_text_only_response():
    response = _make_response([types.Part.from_text(text="hello world")])
    result = _parse_gemini_response(response)
    assert result.stop_reason == "end_turn"
    assert result.content == [{"type": "text", "text": "hello world"}]


def test_parse_function_call_response():
    response = _make_response(
        [types.Part.from_function_call(name="read_file", args={"path": "a.py"})]
    )
    result = _parse_gemini_response(response)
    assert result.stop_reason == "tool_use"
    assert len(result.content) == 1
    block = result.content[0]
    assert block["type"] == "tool_use"
    assert block["name"] == "read_file"
    assert block["input"] == {"path": "a.py"}
    assert block["id"]  # synthesized id is non-empty


def test_parse_usage_metadata():
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=100, candidates_token_count=50
    )
    response = _make_response([types.Part.from_text(text="hi")], usage=usage)
    result = _parse_gemini_response(response)
    assert result.usage == {"input_tokens": 100, "output_tokens": 50}


def test_parse_missing_usage_metadata_does_not_crash():
    response = _make_response([types.Part.from_text(text="hi")], usage=None)
    result = _parse_gemini_response(response)
    assert result.usage == {}


# ---------------------------------------------------------------------------
# _classify_gemini_error
# ---------------------------------------------------------------------------


def _make_api_error(code: int, message: str = "error") -> genai_errors.APIError:
    return genai_errors.APIError(code=code, response_json={"error": {"message": message}})


def test_401_maps_to_auth_error():
    assert isinstance(_classify_gemini_error(_make_api_error(401)), MazuAuthError)


def test_403_maps_to_auth_error():
    assert isinstance(_classify_gemini_error(_make_api_error(403)), MazuAuthError)


def test_429_maps_to_rate_limit_error():
    assert isinstance(_classify_gemini_error(_make_api_error(429)), MazuRateLimitError)


def test_500_maps_to_transient_error():
    assert isinstance(_classify_gemini_error(_make_api_error(500)), MazuTransientError)


def test_context_length_hint_in_message_maps_to_context_length_error():
    err = _make_api_error(400, "the request exceeds the maximum context length allowed")
    assert isinstance(_classify_gemini_error(err), MazuContextLengthError)


def test_non_api_error_still_wrapped():
    from mazu.llm.errors import MazuAPIError

    result = _classify_gemini_error(RuntimeError("something else"))
    assert isinstance(result, MazuAPIError)
    assert not isinstance(result, (MazuAuthError, MazuRateLimitError, MazuTransientError))


# ---------------------------------------------------------------------------
# GeminiProvider._get_client()
# ---------------------------------------------------------------------------


def test_missing_package_raises_clean_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-fake")
    monkeypatch.setitem(sys.modules, "google.genai", None)
    monkeypatch.setitem(sys.modules, "google", None)
    provider = GeminiProvider()
    with pytest.raises(MazuAuthError, match="google-genai package isn't installed"):
        provider._get_client()


def test_missing_api_key_raises_clean_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = GeminiProvider()
    with pytest.raises(MazuAuthError, match="GEMINI_API_KEY"):
        provider._get_client()


def test_non_ascii_key_raises_clean_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sənin-key-in")
    provider = GeminiProvider()
    with pytest.raises(MazuAuthError, match="non-ASCII"):
        provider._get_client()


# ---------------------------------------------------------------------------
# GeminiProvider.run_turn / run_forced_tool (mocked client, real response shapes)
# ---------------------------------------------------------------------------


def test_run_turn_returns_parsed_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-fake")
    provider = GeminiProvider()
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _make_response(
        [types.Part.from_text(text="hello")]
    )
    provider._client = fake_client

    response = provider.run_turn([{"role": "user", "content": "hi"}], "sys", [], "gemini-2.0-flash")

    assert response.stop_reason == "end_turn"
    assert response.content == [{"type": "text", "text": "hello"}]


def test_run_turn_wraps_sdk_errors(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-fake")
    provider = GeminiProvider()
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = _make_api_error(401)
    provider._client = fake_client

    with pytest.raises(MazuAuthError):
        provider.run_turn([{"role": "user", "content": "hi"}], "sys", [], "gemini-2.0-flash")


def test_run_forced_tool_extracts_matching_call(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-fake")
    provider = GeminiProvider()
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _make_response(
        [types.Part.from_function_call(name="my_tool", args={"x": 1})]
    )
    provider._client = fake_client

    tool = {"name": "my_tool", "description": "d", "input_schema": {}}
    result = provider.run_forced_tool([{"role": "user", "content": "hi"}], "sys", tool, "gemini-2.0-flash")

    assert result == {"x": 1}


# ---------------------------------------------------------------------------
# run_turn_stream: must use the Provider base-class fallback, NOT a real
# streaming implementation (deliberate scope decision -- see the class
# docstring in gemini_provider.py for why).
# ---------------------------------------------------------------------------


def test_stream_uses_base_class_fallback_not_overridden():
    assert "run_turn_stream" not in GeminiProvider.__dict__


def test_stream_delivers_full_text_via_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-fake")
    provider = GeminiProvider()
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _make_response(
        [types.Part.from_text(text="full answer")]
    )
    provider._client = fake_client

    received = []
    response = provider.run_turn_stream(
        [{"role": "user", "content": "hi"}], "sys", [], "gemini-2.0-flash", on_delta=received.append
    )

    assert received == ["full answer"]
    assert response.stop_reason == "end_turn"
