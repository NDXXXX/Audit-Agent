"""Shared helper for executing LangChain structured-tool calls."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain_core.tools import StructuredTool


def execute_tool(
    call: Mapping[str, Any],
    *,
    tools_by_name: Mapping[str, StructuredTool],
) -> Any:
    """Execute one model tool call and convert failures into tool results."""

    name = str(call.get("name", ""))
    tool = tools_by_name.get(name)
    if tool is None:
        return {"error": f"Unknown tool: {name}"}

    args = call.get("args") or {}
    if not isinstance(args, Mapping):
        return {"error": f"Invalid arguments for {name}: expected an object"}

    try:
        return tool.invoke(dict(args))
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Shared helpers (used by agents and graph nodes)
# ---------------------------------------------------------------------------

EventWriter = Callable[[dict[str, Any]], Any]


def _content_to_text(content: Any) -> str:
    """Extract a plain string from LangChain message content blocks."""

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


def _record_event(
    event: dict[str, Any],
    *,
    writer: EventWriter | None,
    events: list[dict[str, Any]],
) -> None:
    """Append an event to the local list and emit it via the stream writer."""
    events.append(event)
    if writer is not None:
        writer(event)
