"""Shared helper for executing LangChain structured-tool calls."""

from __future__ import annotations

from collections.abc import Mapping
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
