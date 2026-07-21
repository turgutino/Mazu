from mazu.llm.providers.openai_compatible import OpenAICompatibleProvider


class LocalProvider(OpenAICompatibleProvider):
    """Any OpenAI-compatible server running on the user's own machine -- LM Studio,
    Ollama, llama.cpp's server, etc. No cloud egress, no API key, no per-token cost.
    base_url is resolved lazily (on first real use, not at import time) since
    client.py's _PROVIDERS dict is built eagerly before config-loading has
    necessarily happened yet.
    """

    def __init__(self):
        super().__init__(
            base_url=None,
            api_key_env="MAZU_LOCAL_API_KEY",
            provider_label="local:",
            requires_api_key=False,
        )

    def _get_client(self):
        if self._client is None:
            from mazu.config import local_base_url

            self.base_url = local_base_url()
        return super()._get_client()
