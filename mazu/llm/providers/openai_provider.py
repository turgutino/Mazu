from mazu.llm.providers.openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    def __init__(self):
        super().__init__(base_url=None, api_key_env="OPENAI_API_KEY", provider_label="openai:")
