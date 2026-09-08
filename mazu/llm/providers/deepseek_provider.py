from mazu.llm.providers.openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    def __init__(self):
        super().__init__(
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            provider_label="deepseek:",
        )

    def supports_vision_for(self, model_name: str) -> bool:
        # DeepSeek's naming convention marks its vision-capable models explicitly
        # (e.g. "deepseek-v4-flash-vision-exp", seen live in its own /v1/models
        # response) -- its mainline chat/reasoner models are text-only. Real bug
        # found live: attaching an image while chatting with a plain
        # deepseek:deepseek-chat session produced a raw provider error instead of
        # any upfront warning; this is a best-effort, UI-only signal (see
        # Provider.supports_vision_for's docstring), not a request gate.
        return "vision" in model_name.lower()
