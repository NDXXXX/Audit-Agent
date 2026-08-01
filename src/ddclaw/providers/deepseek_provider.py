"""DeepSeek chat-model factory for ddclaw."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek


def create_model(
    model: str | None = None,
    **kwargs: Any,
) -> ChatDeepSeek:
    """Create a ``ChatDeepSeek`` instance using values loaded from ``.env``."""

    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Add it to .env or the process environment."
        )

    model_name = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    kwargs.setdefault(
        "api_base",
        os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
    )
    kwargs.setdefault("temperature", 0)
    kwargs.setdefault("timeout", 120)
    kwargs.setdefault("max_retries", 2)
    return ChatDeepSeek(model=model_name, api_key=api_key, **kwargs)
