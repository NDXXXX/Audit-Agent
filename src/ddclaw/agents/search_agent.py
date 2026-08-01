"""Focused ReAct agent for web research."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ddclaw.providers import create_model
from ddclaw.tools.execution import execute_tool
from ddclaw.tools.web_search_tool import WebSearchTool

SEARCH_AGENT_PROMPT = """You are searchAgent, a focused research specialist.

Your only external capability is WebSearchTool. Search for reliable information
needed by the planner and codeAgent.

Rules:
- Use WebSearchTool for factual research.
- Prefer official or encyclopedia-style sources when available.
- Return a concise research summary and list the useful source URLs.
- Do not write files or produce application code.
"""

EventWriter = Callable[[dict[str, Any]], Any]


def run_search_agent(
    state: Mapping[str, Any],
    instruction: str,
    *,
    writer: EventWriter | None = None,
    max_loops: int = 4,
) -> dict[str, Any]:
    """Run the search specialist and return its accumulated research."""

    if max_loops < 1:
        raise ValueError("max_loops must be greater than or equal to 1")

    web_search = WebSearchTool()
    search_tool = web_search.as_structured_tool()
    tools_by_name = {search_tool.name: search_tool}
    agent = create_model().bind_tools([search_tool])

    request = {
        "task": state.get("task", ""),
        "instruction": instruction,
        "research_notes": state.get("research_notes", ""),
    }
    messages: list[BaseMessage] = [
        SystemMessage(content=SEARCH_AGENT_PROMPT),
        HumanMessage(
            content=json.dumps(request, ensure_ascii=False, default=str),
        ),
    ]
    queries: list[str] = []
    sources: list[str] = []
    answers: list[str] = []
    tool_events: list[dict[str, Any]] = []
    summary = ""

    for _ in range(max_loops):
        response = agent.invoke(messages)
        messages.append(response)
        response_text = _content_to_text(response.content)
        if response_text:
            summary = response_text

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break

        for call in tool_calls:
            name = str(call.get("name", ""))
            args = call.get("args") or {}
            call_event = {
                "type": "tool_call",
                "agent": "searchAgent",
                "name": name,
                "args": args,
            }
            _record_event(call_event, writer=writer, events=tool_events)

            result = execute_tool(call, tools_by_name=tools_by_name)
            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=str(call.get("id", "")),
                )
            )

            result_event = _search_result_event(result)
            result_event["agent"] = "searchAgent"
            _record_event(result_event, writer=writer, events=tool_events)
            _collect_search_result(
                result,
                fallback_query=(
                    str(args.get("query", ""))
                    if isinstance(args, Mapping)
                    else ""
                ),
                queries=queries,
                sources=sources,
                answers=answers,
            )

    if not summary:
        summary = _fallback_summary(answers, sources, tool_events)

    return {
        "ok": True,
        "summary": summary,
        "queries": queries,
        "sources": sources,
        "messages": messages,
        "tool_events": tool_events,
    }


def _record_event(
    event: dict[str, Any],
    *,
    writer: EventWriter | None,
    events: list[dict[str, Any]],
) -> None:
    events.append(event)
    if writer is not None:
        writer(event)


def _search_result_event(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return {
            "type": "search_results",
            **dict(result),
        }
    return {
        "type": "search_results",
        "ok": False,
        "error": "WebSearchTool returned an invalid result",
        "result": result,
    }


def _collect_search_result(
    result: Any,
    *,
    fallback_query: str,
    queries: list[str],
    sources: list[str],
    answers: list[str],
) -> None:
    if not isinstance(result, Mapping):
        return

    query = str(result.get("query") or fallback_query).strip()
    if query and query not in queries:
        queries.append(query)

    answer = result.get("answer")
    if isinstance(answer, str) and answer.strip() and answer not in answers:
        answers.append(answer.strip())

    results = result.get("results", [])
    if not isinstance(results, list):
        return
    for item in results:
        if not isinstance(item, Mapping):
            continue
        url = item.get("url")
        if isinstance(url, str) and url and url not in sources:
            sources.append(url)


def _fallback_summary(
    answers: list[str],
    sources: list[str],
    tool_events: list[dict[str, Any]],
) -> str:
    if answers:
        return "\n\n".join(answers)
    if sources:
        return f"Research collected {len(sources)} useful source(s)."

    errors = [
        str(event.get("error"))
        for event in tool_events
        if event.get("type") == "search_results" and event.get("error")
    ]
    if errors:
        return f"Search could not be completed: {errors[-1]}"
    return "No web research was produced."


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)
