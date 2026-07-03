from mazu.llm.providers.openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    def __init__(self):
        super().__init__(
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            provider_label="deepseek:",
        )
