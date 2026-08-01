"""Tavily-backed web search exposed as a LangChain structured tool."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tavily import TavilyClient


class WebSearchInput(BaseModel):
    """Arguments accepted by the web-search tool."""

    query: str = Field(
        min_length=1,
        description="Focused factual web-search query.",
    )


class _TavilyClient(Protocol):
    def search(self, query: str, **kwargs: Any) -> Mapping[str, Any]: ...


class WebSearchTool:
    """Search the web with Tavily and return a stable result shape."""

    name = "web_search"

    def __init__(
        self,
        *,
        client_factory: Callable[[str], _TavilyClient] | None = None,
    ) -> None:
        self._client_factory = client_factory or _create_tavily_client

    def __call__(self, query: str) -> dict[str, Any]:
        return self.run(query=query)

    def run(self, query: str) -> dict[str, Any]:
        load_dotenv()
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {
                "ok": False,
                "error": "missing TAVILY_API_KEY",
            }

        normalized_query = query.strip()
        if not normalized_query:
            return {
                "ok": False,
                "query": query,
                "error": "query must not be empty",
            }

        try:
            client = self._client_factory(api_key)
            response = client.search(
                query=normalized_query,
                include_answer=True,
            )
        except Exception as exc:
            return {
                "ok": False,
                "query": normalized_query,
                "error": f"{type(exc).__name__}: {exc}",
            }

        if not isinstance(response, Mapping):
            return {
                "ok": False,
                "query": normalized_query,
                "error": "Tavily returned an invalid response",
            }

        results: list[dict[str, Any]] = []
        raw_results = response.get("results", [])
        if isinstance(raw_results, list):
            for item in raw_results:
                if not isinstance(item, Mapping):
                    continue
                results.append(
                    {
                        "title": _as_text(item.get("title")),
                        "url": _as_text(item.get("url")),
                        "content": _as_text(item.get("content")),
                        "score": _as_score(item.get("score")),
                    }
                )

        return {
            "ok": True,
            "query": _as_text(response.get("query")) or normalized_query,
            "answer": _as_text(response.get("answer")),
            "results": results,
        }

    def as_structured_tool(self) -> StructuredTool:
        """Return the LangChain tool passed to ``model.bind_tools``."""

        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description=(
                "Search the public web for reliable factual information with "
                "Tavily. Returns an answer and ranked source snippets."
            ),
            args_schema=WebSearchInput,
        )


def _create_tavily_client(api_key: str) -> _TavilyClient:
    return TavilyClient(api_key=api_key)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _as_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
