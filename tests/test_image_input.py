"""Tests for image (vision) input support across providers. Mazu's canonical
message format already uses Anthropic's own image block shape
(`{"type": "image", "source": {"type": "base64", "media_type": ..., "data": ...}}`),
so AnthropicProvider needs no conversion at all -- only the OpenAI-compatible and
Gemini converters need a new branch to translate that shape into their own wire
format.
"""

import base64

from mazu.llm.providers.gemini_provider import _to_gemini_contents
from mazu.llm.providers.openai_compatible import _to_openai_messages

_FAKE_PNG_BASE64 = base64.b64encode(b"not a real png, just test bytes").decode("ascii")

_IMAGE_BLOCK = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": _FAKE_PNG_BASE64},
}

# ---------------------------------------------------------------------------
# OpenAI-compatible (shared by OpenAI, DeepSeek, local)
# ---------------------------------------------------------------------------


def test_openai_image_only_becomes_image_url_part():
    messages = [{"role": "user", "content": [_IMAGE_BLOCK]}]
    result = _to_openai_messages("sys", messages)

    user_msg = result[-1]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_FAKE_PNG_BASE64}"}}
    ]


def test_openai_image_and_text_stay_in_one_message():
    messages = [
        {
            "role": "user",
            "content": [_IMAGE_BLOCK, {"type": "text", "text": "what is this?"}],
        }
    ]
    result = _to_openai_messages("sys", messages)

    # system message + exactly one combined user message -- image and its
    # caption must not be split into separate turns.
    assert len(result) == 2
    user_msg = result[1]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_FAKE_PNG_BASE64}"}},
        {"type": "text", "text": "what is this?"},
    ]


def test_openai_single_text_block_still_becomes_bare_string():
    # Backward-compat: a lone text block (the common, pre-image case) keeps the
    # old bare-string shape rather than becoming a single-item array -- some
    # OpenAI-compatible servers are stricter about array content than OpenAI itself.
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    result = _to_openai_messages("sys", messages)
    assert result[-1] == {"role": "user", "content": "hello"}


def test_openai_tool_result_and_image_turn_both_present():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "file contents"},
            ],
        },
        {"role": "user", "content": [_IMAGE_BLOCK]},
    ]
    result = _to_openai_messages("sys", messages)
    roles = [m["role"] for m in result]
    assert roles == ["system", "tool", "user"]
    assert result[-1]["content"][0]["type"] == "image_url"


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


def test_gemini_image_becomes_inline_bytes_part():
    contents = _to_gemini_contents([{"role": "user", "content": [_IMAGE_BLOCK]}])

    assert len(contents) == 1
    part = contents[0].parts[0]
    assert part.inline_data.mime_type == "image/png"
    assert part.inline_data.data == base64.b64decode(_FAKE_PNG_BASE64)


def test_gemini_image_and_text_share_one_content():
    contents = _to_gemini_contents(
        [{"role": "user", "content": [_IMAGE_BLOCK, {"type": "text", "text": "describe it"}]}]
    )

    assert len(contents) == 1
    assert contents[0].role == "user"
    assert len(contents[0].parts) == 2
    assert contents[0].parts[0].inline_data.mime_type == "image/png"
    assert contents[0].parts[1].text == "describe it"
