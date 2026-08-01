import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from ddclaw.agents import search_agent as search_agent_module
from ddclaw.tools import web_search_tool as web_search_module
from ddclaw.tools.web_search_tool import WebSearchInput, WebSearchTool


class FakeTavilyClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"query": query, **kwargs})
        return self.response


def test_web_search_tool_returns_missing_key_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_search_module, "load_dotenv", lambda: None)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    factory_called = False

    def client_factory(api_key: str) -> FakeTavilyClient:
        nonlocal factory_called
        factory_called = True
        raise AssertionError(f"unexpected key: {api_key}")

    result = WebSearchTool(client_factory=client_factory).run("Python docs")

    assert result == {
        "ok": False,
        "error": "missing TAVILY_API_KEY",
    }
    assert factory_called is False


def test_web_search_tool_normalizes_tavily_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_search_module, "load_dotenv", lambda: None)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    client = FakeTavilyClient(
        {
            "query": "official Python documentation",
            "answer": "Python documentation is published at docs.python.org.",
            "results": [
                {
                    "title": "Python 3 Documentation",
                    "url": "https://docs.python.org/3/",
                    "content": "Official Python language and library docs.",
                    "score": 0.99,
                    "raw_content": "not exposed",
                },
                {
                    "title": None,
                    "url": "https://www.python.org/",
                    "content": None,
                    "score": "0.75",
                },
                "invalid result",
            ],
        }
    )
    received_keys: list[str] = []

    def client_factory(api_key: str) -> FakeTavilyClient:
        received_keys.append(api_key)
        return client

    result = WebSearchTool(client_factory=client_factory).run(
        "  official Python documentation  "
    )

    assert received_keys == ["tvly-test"]
    assert client.calls == [
        {
            "query": "official Python documentation",
            "include_answer": True,
        }
    ]
    assert result == {
        "ok": True,
        "query": "official Python documentation",
        "answer": "Python documentation is published at docs.python.org.",
        "results": [
            {
                "title": "Python 3 Documentation",
                "url": "https://docs.python.org/3/",
                "content": "Official Python language and library docs.",
                "score": 0.99,
            },
            {
                "title": "",
                "url": "https://www.python.org/",
                "content": "",
                "score": 0.75,
            },
        ],
    }


class FakeSearchTool:
    name = "web_search"

    def run(self, query: str) -> dict[str, Any]:
        assert query == "official Python testing guidance"
        return {
            "ok": True,
            "query": query,
            "answer": "Python recommends automated tests for reliable software.",
            "results": [
                {
                    "title": "Python testing tools",
                    "url": "https://docs.python.org/3/library/unittest.html",
                    "content": "The unittest testing framework.",
                    "score": 0.95,
                }
            ],
        }

    def as_structured_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description="Search the web.",
            args_schema=WebSearchInput,
        )


class FakeBoundSearchModel:
    def __init__(self) -> None:
        self.responses = iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "web_search",
                            "args": {
                                "query": "official Python testing guidance",
                            },
                            "id": "search-call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content=(
                        "Python's official documentation describes its "
                        "standard testing framework.\n"
                        "Source: https://docs.python.org/3/library/unittest.html"
                    )
                ),
            ]
        )
        self.invocations: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.invocations.append(list(messages))
        return next(self.responses)


class FakeSearchModel:
    def __init__(self, bound: FakeBoundSearchModel) -> None:
        self.bound = bound
        self.tools: list[StructuredTool] | None = None

    def bind_tools(self, tools: list[StructuredTool]) -> FakeBoundSearchModel:
        self.tools = tools
        return self.bound


def test_run_search_agent_collects_research_and_streams_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = FakeBoundSearchModel()
    model = FakeSearchModel(bound)
    monkeypatch.setattr(search_agent_module, "create_model", lambda: model)
    monkeypatch.setattr(search_agent_module, "WebSearchTool", FakeSearchTool)
    written_events: list[dict[str, Any]] = []

    result = search_agent_module.run_search_agent(
        {
            "task": "Build a Python application",
            "research_notes": "Prefer primary sources.",
        },
        "Find official testing guidance",
        writer=written_events.append,
    )

    assert result["ok"] is True
    assert result["queries"] == ["official Python testing guidance"]
    assert result["sources"] == [
        "https://docs.python.org/3/library/unittest.html"
    ]
    assert "official documentation" in result["summary"]
    assert result["tool_events"] == written_events
    assert [event["type"] for event in written_events] == [
        "tool_call",
        "search_results",
    ]
    assert model.tools is not None
    assert [tool.name for tool in model.tools] == ["web_search"]

    first_messages = bound.invocations[0]
    assert isinstance(first_messages[0], SystemMessage)
    assert first_messages[0].content == search_agent_module.SEARCH_AGENT_PROMPT
    assert isinstance(first_messages[1], HumanMessage)
    request = json.loads(first_messages[1].content)
    assert request == {
        "task": "Build a Python application",
        "instruction": "Find official testing guidance",
        "research_notes": "Prefer primary sources.",
    }

    second_messages = bound.invocations[1]
    assert isinstance(second_messages[-1], ToolMessage)
    tool_result = json.loads(second_messages[-1].content)
    assert tool_result["ok"] is True
    assert second_messages[-1].tool_call_id == "search-call-1"


def test_run_search_agent_validates_max_loops() -> None:
    with pytest.raises(ValueError, match="max_loops"):
        search_agent_module.run_search_agent(
            {},
            "research",
            max_loops=0,
        )
