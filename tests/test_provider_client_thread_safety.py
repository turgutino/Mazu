"""Regression test for a real crash found while building mazu-windows: several
QThreads (Chat/Run/Explore/Council/Curator/model-listing) all call into the same
Provider instance (mazu.llm.client._PROVIDERS is a module-level singleton), so two
threads could both see `self._client is None` on their very first call and both
construct the SDK client concurrently -- reproduced live as a Windows native crash
(0xc0000374, heap corruption) inside concurrent `ssl.create_default_context()`
calls. After the first successful construction later calls are just safe attribute
reads, so the danger window is specifically "first call to a given provider from
more than one thread at once."

These tests hammer that exact window with a fresh, never-yet-used Provider
instance and assert exactly one client object is ever built and cached, for each
of the four provider classes that lazily construct an SDK client.
"""

import threading

import pytest

from mazu.llm.providers.anthropic_provider import AnthropicProvider
from mazu.llm.providers.deepseek_provider import DeepSeekProvider
from mazu.llm.providers.gemini_provider import GeminiProvider
from mazu.llm.providers.local_provider import LocalProvider
from mazu.llm.providers.openai_provider import OpenAIProvider

N_THREADS = 32


def _hammer_get_client(provider) -> list:
    """Runs `provider._get_client()` from many threads at once, all starting as
    close together as possible via a barrier, and returns every client object
    obtained (one per thread)."""
    barrier = threading.Barrier(N_THREADS)
    results: list = [None] * N_THREADS
    errors: list = []

    def worker(i):
        barrier.wait()
        try:
            results[i] = provider._get_client()
        except Exception as e:  # pragma: no cover - only fires on a real bug
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise errors[0]
    return results


def test_anthropic_get_client_builds_exactly_one_client_under_concurrency(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    provider = AnthropicProvider()

    results = _hammer_get_client(provider)

    assert all(r is provider._client for r in results)
    assert len({id(r) for r in results}) == 1


def test_gemini_get_client_builds_exactly_one_client_under_concurrency(monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = GeminiProvider()

    results = _hammer_get_client(provider)

    assert all(r is provider._client for r in results)
    assert len({id(r) for r in results}) == 1


def test_openai_get_client_builds_exactly_one_client_under_concurrency(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    provider = OpenAIProvider()

    results = _hammer_get_client(provider)

    assert all(r is provider._client for r in results)
    assert len({id(r) for r in results}) == 1


def test_deepseek_get_client_builds_exactly_one_client_under_concurrency(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    provider = DeepSeekProvider()

    results = _hammer_get_client(provider)

    assert all(r is provider._client for r in results)
    assert len({id(r) for r in results}) == 1


def test_local_get_client_builds_exactly_one_client_under_concurrency(monkeypatch):
    monkeypatch.setattr("mazu.config.local_base_url", lambda: "http://localhost:1234/v1")
    provider = LocalProvider()

    results = _hammer_get_client(provider)

    assert all(r is provider._client for r in results)
    assert len({id(r) for r in results}) == 1
