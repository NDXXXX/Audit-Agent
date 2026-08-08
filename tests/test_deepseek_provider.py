from typing import Any

import pytest

from audit_agent.providers import deepseek_provider


class FakeChatDeepSeek:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def test_create_model_uses_deepseek_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(deepseek_provider, "load_dotenv", lambda: False)
    monkeypatch.setattr(deepseek_provider, "ChatDeepSeek", FakeChatDeepSeek)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_BASE", "https://example.test")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    model = deepseek_provider.create_model()

    assert isinstance(model, FakeChatDeepSeek)
    assert model.kwargs == {
        "model": "deepseek-v4-flash",
        "api_key": "test-key",
        "api_base": "https://example.test",
        "temperature": 0,
        "timeout": 120,
        "max_retries": 2,
    }


def test_create_model_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(deepseek_provider, "load_dotenv", lambda: False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        deepseek_provider.create_model()


def test_create_model_allows_explicit_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(deepseek_provider, "load_dotenv", lambda: False)
    monkeypatch.setattr(deepseek_provider, "ChatDeepSeek", FakeChatDeepSeek)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    model = deepseek_provider.create_model(
        model="deepseek-v4-pro",
        timeout=30,
        max_retries=4,
    )

    assert model.kwargs["model"] == "deepseek-v4-pro"
    assert model.kwargs["timeout"] == 30
    assert model.kwargs["max_retries"] == 4
