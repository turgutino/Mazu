from mazu.llm.providers.openai_compatible import OpenAICompatibleProvider

# OpenAI's /v1/models response is one flat list covering its entire product
# surface, not just chat -- these substrings mark ids that are never usable as a
# chat-completions `model=` value (embeddings, audio, image, moderation, and
# legacy plain-completion models), so list_models() can filter them out. Not
# exhaustive by design (see _is_chat_model's docstring): an unrecognized future
# non-chat family shows up in the picker rather than being silently guessed away.
_NON_CHAT_MARKERS = (
    "embedding", "whisper", "tts", "dall-e", "moderation", "audio",
    "davinci", "babbage", "curie", "ada-", "image", "realtime",
    "transcribe", "computer-use", "text-moderation",
)


class OpenAIProvider(OpenAICompatibleProvider):
    # Only OpenAI's own reasoning-model families (o-series, gpt-5) accept
    # `reasoning_effort` -- a non-reasoning model (e.g. gpt-4o) rejects it with a
    # 400. supports_reasoning_effort stays True at the class level because
    # run_turn/run_turn_stream's actual behavior (send it whenever asked, let a
    # non-reasoning model's own 400 surface as a normal MazuAPIError) is
    # unchanged -- narrowing THAT by model name would need a real allowlist kept
    # in lockstep with the request path, worth avoiding for the same staleness
    # reason _is_chat_model's docstring gives. supports_reasoning_effort_for()
    # below is a SEPARATE, best-effort, UI-only signal (mazu-web's effort-chip
    # disabling) -- a naming-convention heuristic, not an allowlist: a brand-new
    # reasoning family this doesn't recognize just means the chip stays enabled
    # for it (fails open), it never blocks or changes an actual request.
    supports_reasoning_effort = True

    _REASONING_MODEL_PREFIXES = ("o1", "o3", "o4")

    def __init__(self):
        super().__init__(base_url=None, api_key_env="OPENAI_API_KEY", provider_label="openai:")

    def supports_reasoning_effort_for(self, model_name: str) -> bool:
        lowered = model_name.lower()
        return lowered.startswith(self._REASONING_MODEL_PREFIXES) or "gpt-5" in lowered

    def _is_chat_model(self, model_id: str) -> bool:
        """A deliberately conservative substring denylist, not an allowlist: new
        chat-capable model families ship often enough that an allowlist would need
        constant upkeep and would hide a brand-new model from the picker the day it
        launches. Missing a new non-chat family here just means it appears in the
        list alongside real chat models -- annoying, not broken -- which is the
        safer failure direction.
        """
        lowered = model_id.lower()
        return not any(marker in lowered for marker in _NON_CHAT_MARKERS)
