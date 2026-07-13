import sys
from unittest.mock import MagicMock

import pytest

from mazu.memory.embeddings import (
    cosine_similarity,
    deserialize_embedding,
    embed_text,
    embeddings_available,
    serialize_embedding,
)

# ---------------------------------------------------------------------------
# embeddings_available: must require ALL THREE of the opt-in env var, the key,
# and the package -- missing any one disables the feature entirely.
# ---------------------------------------------------------------------------


def test_unavailable_when_opt_in_flag_not_set(monkeypatch):
    monkeypatch.delenv("MAZU_SEMANTIC_MEMORY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    assert embeddings_available() is False


def test_unavailable_when_key_not_set(monkeypatch):
    monkeypatch.setenv("MAZU_SEMANTIC_MEMORY", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert embeddings_available() is False


def test_unavailable_when_package_missing(monkeypatch):
    monkeypatch.setenv("MAZU_SEMANTIC_MEMORY", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setitem(sys.modules, "openai", None)
    assert embeddings_available() is False


def test_available_when_all_three_conditions_met(monkeypatch):
    monkeypatch.setenv("MAZU_SEMANTIC_MEMORY", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    assert embeddings_available() is True


def test_flag_accepts_true_and_yes(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    for value in ("1", "true", "TRUE", "yes", "Yes"):
        monkeypatch.setenv("MAZU_SEMANTIC_MEMORY", value)
        assert embeddings_available() is True
    for value in ("0", "false", "no", ""):
        monkeypatch.setenv("MAZU_SEMANTIC_MEMORY", value)
        assert embeddings_available() is False


# ---------------------------------------------------------------------------
# embed_text (mocked)
# ---------------------------------------------------------------------------


def test_embed_text_returns_none_when_not_available(monkeypatch):
    monkeypatch.delenv("MAZU_SEMANTIC_MEMORY", raising=False)
    assert embed_text("some text") is None


def test_embed_text_returns_none_for_empty_string(monkeypatch):
    monkeypatch.setenv("MAZU_SEMANTIC_MEMORY", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    assert embed_text("   ") is None


def test_embed_text_calls_api_and_returns_vector(monkeypatch):
    monkeypatch.setenv("MAZU_SEMANTIC_MEMORY", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    fake_client = MagicMock()
    fake_embedding_obj = MagicMock()
    fake_embedding_obj.embedding = [0.1, 0.2, 0.3]
    fake_response = MagicMock()
    fake_response.data = [fake_embedding_obj]
    fake_client.embeddings.create.return_value = fake_response

    monkeypatch.setattr("openai.OpenAI", lambda: fake_client)

    result = embed_text("we use PostgreSQL")
    assert result == [0.1, 0.2, 0.3]
    fake_client.embeddings.create.assert_called_once()


def test_embed_text_returns_none_on_api_failure(monkeypatch):
    monkeypatch.setenv("MAZU_SEMANTIC_MEMORY", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    fake_client = MagicMock()
    fake_client.embeddings.create.side_effect = RuntimeError("network error")
    monkeypatch.setattr("openai.OpenAI", lambda: fake_client)

    assert embed_text("text") is None


# ---------------------------------------------------------------------------
# serialize/deserialize
# ---------------------------------------------------------------------------


def test_serialize_deserialize_roundtrip():
    vector = [0.1, -0.2, 0.3333333]
    assert deserialize_embedding(serialize_embedding(vector)) == vector


def test_deserialize_none_returns_none():
    assert deserialize_embedding(None) is None


def test_deserialize_empty_string_returns_none():
    assert deserialize_embedding("") is None


def test_deserialize_malformed_json_returns_none():
    assert deserialize_embedding("{not valid json") is None


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


def test_identical_vectors_similarity_is_one():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_orthogonal_vectors_similarity_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors_similarity_is_negative_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_mismatched_length_returns_zero():
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_empty_vectors_return_zero():
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0], []) == 0.0


def test_zero_vector_returns_zero_not_crash():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# LIVE test: a real OpenAI embedding call, proving the module works against the
# actual API (not just its documented shape) and that cosine similarity genuinely
# reflects semantic closeness, not just "returns floats". Skipped automatically
# when OPENAI_API_KEY isn't set (e.g. in CI), so this never blocks the normal suite.
# ---------------------------------------------------------------------------


@pytest.mark.skipif("OPENAI_API_KEY" not in __import__("os").environ, reason="no live OpenAI key")
def test_live_semantic_similarity_beats_unrelated_text(monkeypatch):
    monkeypatch.setenv("MAZU_SEMANTIC_MEMORY", "1")

    e1 = embed_text("we use PostgreSQL for the database")
    e2 = embed_text("the project's database is Postgres")  # same fact, different words
    e3 = embed_text("the cat sat on the mat")  # unrelated

    assert e1 is not None and e2 is not None and e3 is not None
    related_score = cosine_similarity(e1, e2)
    unrelated_score = cosine_similarity(e1, e3)

    assert related_score > unrelated_score
    assert related_score > 0.7  # same fact restated should be very close
