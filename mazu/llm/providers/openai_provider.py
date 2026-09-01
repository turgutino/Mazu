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
    # 400. Deliberately not narrowed further by model name here (see
    # _is_chat_model's docstring for the same reasoning): a hardcoded
    # reasoning-vs-non-reasoning model list would go stale the day OpenAI ships a
    # new family. Picking an effort level with a non-reasoning model surfaces as a
    # normal MazuAPIError from the provider, same as any other invalid request.
    supports_reasoning_effort = True

    def __init__(self):
        super().__init__(base_url=None, api_key_env="OPENAI_API_KEY", provider_label="openai:")

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
