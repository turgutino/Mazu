"""Optional semantic layer on top of BM25 (mazu/memory/bm25.py). Deliberately opt-in
via MAZU_SEMANTIC_MEMORY, not auto-activated just because an OpenAI key happens to be
set -- someone using deepseek: as their main model but who also has an OpenAI key
around (e.g. for council mode) shouldn't have `remember` silently start making
background OpenAI calls they didn't ask for. Every function here degrades to "not
available" / None rather than raising, so a missing key, missing package, or a failed
API call never breaks memory writes or retrieval -- BM25 alone keeps working exactly
as it did before this module existed.
"""

import json
import math
import os

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_API_KEY_ENV = "OPENAI_API_KEY"
_ENABLE_ENV = "MAZU_SEMANTIC_MEMORY"


def embeddings_available() -> bool:
    if os.environ.get(_ENABLE_ENV, "").strip().lower() not in ("1", "true", "yes"):
        return False
    if not os.environ.get(EMBEDDING_API_KEY_ENV):
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


def embed_text(text: str) -> list[float] | None:
    """Returns an embedding vector for `text`, or None if semantic search isn't
    enabled/available, or if the API call itself fails for any reason. Broad
    exception handling is deliberate here: this is always an optional enhancement,
    never a hard dependency, so any failure should silently fall back to BM25-only
    behavior rather than surface an error for something the user didn't explicitly
    request in the moment (they opted into the feature once, via the env var, not
    into every individual call succeeding).
    """
    if not embeddings_available() or not text.strip():
        return None
    try:
        import openai

        client = openai.OpenAI()
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return response.data[0].embedding
    except Exception:
        return None


def serialize_embedding(vector: list[float]) -> str:
    return json.dumps(vector)


def deserialize_embedding(raw) -> list[float] | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure Python, no numpy -- same zero-heavy-dependency approach memory/bm25.py
    already uses for the same reason (keep the package small and installable
    everywhere without a compiled-extension dependency).
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
