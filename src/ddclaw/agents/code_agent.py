"""Focused ReAct agent for workspace-scoped implementation work."""

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

from ddclaw.core.state import RuntimeState
from ddclaw.graph.memory import (
    build_layered_memory,
    format_layered_memory_for_prompt,
    memory_event,
)
from ddclaw.providers import create_model
from ddclaw.tools.execution import execute_tool
from ddclaw.tools.registry import build_tools
from ddclaw.tools.todo_tools import TodoUpdateTool

CODE_AGENT_PROMPT = """You are codeAgent, a focused implementation specialist.

You implement the planner's instruction inside the workspace using file and
shell tools.

Rules:
- You must update todo progress explicitly.
- Before starting a todo, call TodoUpdateTool with status "in_progress".
- After finishing that todo, call TodoUpdateTool with status "completed".
- If a todo is impossible, call TodoUpdateTool with status "blocked" and explain.
- Use FileWriteTool for new files.
- Use FileReadTool before editing existing files.
- Use FileEditTool for focused edits.
- Use BashTool for non-interactive checks.
- Use NotepadAppendTool to record durable findings, decisions, important files,
  blockers, and next-step context that should survive compression.
- Use NotepadReadTool when you need to recover prior notes.
- BashTool already runs inside the workspace. Use relative paths, never "cd /workspace".
- Incorporate research notes and source URLs when the task asks for researched content.
- End with a concise summary of files changed and checks run.
"""

EventWriter = Callable[[dict[str, Any]], Any]


def run_code_agent(
    state: Mapping[str, Any],
    instruction: str,
    *,
    writer: EventWriter | None = None,
    max_loops: int = 10,
) -> dict[str, Any]:
    """Run the implementation specialist and return its updated session state."""

    if max_loops < 1:
        raise ValueError("max_loops must be greater than or equal to 1")

    runtime = _require_runtime(state)
    todo_updater = TodoUpdateTool(state.get("todos", []))
    todo_update_tool = todo_updater.as_structured_tool()
    tools = [*build_tools(runtime), todo_update_tool]
    tools_by_name = {tool.name: tool for tool in tools}
    agent = create_model().bind_tools(tools)

    memory = build_layered_memory_snapshot(state)
    if writer is not None:
        writer(memory_event(memory, node="codeAgent"))
    messages: list[BaseMessage] = [
        SystemMessage(content=CODE_AGENT_PROMPT),
        HumanMessage(content=_code_agent_input(state, instruction, memory)),
    ]
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
            _record_event(
                {
                    "type": "tool_call",
                    "agent": "codeAgent",
                    "name": name,
                    "args": args,
                },
                writer=writer,
                events=tool_events,
            )

            result = execute_tool(call, tools_by_name=tools_by_name)
            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=str(call.get("id", "")),
                )
            )
            _record_event(
                {
                    "type": "tool_result",
                    "agent": "codeAgent",
                    "name": name,
                    "result": result,
                },
                writer=writer,
                events=tool_events,
            )

    if not summary:
        summary = (
            f"codeAgent stopped after {max_loops} loops without a final summary."
        )

    return {
        "ok": True,
        "summary": summary,
        "todos": todo_updater.todos,
        "messages": messages,
        "tool_events": tool_events,
    }


def build_layered_memory_snapshot(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the codeAgent-specific layered-memory snapshot."""

    return build_layered_memory(state, node="codeAgent")


def _build_session_context(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "plan_summary": state.get("plan_summary", ""),
        "todos": state.get("todos", []),
        "acceptance_criteria": state.get("acceptance_criteria", []),
        "verification_commands": state.get("verification_commands", []),
        "research_notes": state.get("research_notes", ""),
        "sources": state.get("sources", []),
        "last_error": state.get("last_error", ""),
    }


def _code_agent_input(
    state: Mapping[str, Any],
    instruction: str,
    memory: Mapping[str, Any],
) -> str:
    """Format the specialist request with its runtime memory snapshot."""

    request = {
        "task": state.get("task", ""),
        "instruction": instruction,
        "session_context": _build_session_context(state),
    }
    return (
        f"{json.dumps(request, ensure_ascii=False, default=str)}"
        "\n\nLayered memory:\n"
        f"{format_layered_memory_for_prompt(memory)}"
    )


def _require_runtime(state: Mapping[str, Any]) -> RuntimeState:
    runtime = state.get("runtime")
    if not isinstance(runtime, RuntimeState):
        raise TypeError("state.runtime must be a RuntimeState")
    return runtime


def _record_event(
    event: dict[str, Any],
    *,
    writer: EventWriter | None,
    events: list[dict[str, Any]],
) -> None:
    events.append(event)
    if writer is not None:
        writer(event)


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
