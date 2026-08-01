"""Runtime-assembled layered memory for DDclaw graph agents."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ddclaw.core.state import RuntimeState
from ddclaw.graph.state import AgentHandoff, LayeredMemory, SourceItem

RULES_LAYER = {
    "scope": "workspace",
    "storage": "internal",
    "rules": [
        "Work inside the current workspace only.",
        "Use paths relative to the workspace; do not prefix paths with workspace/.",
        "Keep durable task context outside the raw messages transcript when possible.",
        "Treat TODO.md as working plan state, NOTEPAD.md as durable notes, and HISTORY_SUMMARY.md as compressed history.",
        "Do not expose memory write tools to agents; layered memory is assembled by the runtime.",
    ],
}

_NOTEPAD_PATH = "NOTEPAD.md"
_HISTORY_PATH = "HISTORY_SUMMARY.md"


def build_layered_memory(
    state: Mapping[str, Any],
    *,
    node: str = "graph",
) -> LayeredMemory:
    """Build the rules, working-memory, and compressed-history layers."""

    runtime = state["runtime"]
    if not isinstance(runtime, RuntimeState):
        raise TypeError("state.runtime must be a RuntimeState")

    notepad = read_notepad(runtime)
    history = read_history_summary(runtime)
    history_summary = str(
        state.get("history_summary")
        or history.get("content", "")
    )

    working_memory = {
        "node": node,
        "task": state.get("task", ""),
        "session_id": state.get("session_id", ""),
        "session_turn": state.get("session_turn", 0),
        "session_context": _short_text(
            state.get("session_context", ""),
            7000,
        ),
        "plan_summary": state.get("plan_summary", ""),
        "todos": state.get("todos", []),
        "acceptance_criteria": state.get("acceptance_criteria", []),
        "verification_commands": state.get("verification_commands", []),
        "research_notes": _short_text(
            state.get("research_notes", ""),
            1600,
        ),
        "sources": _source_prompt_items(state.get("sources", [])),
        "agent_handoffs": _trim_handoffs(
            state.get("agent_handoffs", [])
        ),
        "code_agent_summary": _short_text(
            state.get("code_agent_summary", ""),
            1000,
        ),
        "verifier_summary": _short_text(
            state.get("verifier_summary", ""),
            1000,
        ),
        "last_error": _short_text(state.get("last_error", ""), 1400),
        "attempts": state.get("attempts", 0),
        "max_attempts": state.get("max_attempts", 3),
    }

    compression_events = state.get("compression_events", [])
    if not isinstance(compression_events, Sequence) or isinstance(
        compression_events,
        (str, bytes),
    ):
        compression_events = []
    history_summary_store = {
        "history_path": _HISTORY_PATH,
        "history_exists": history.get("exists", False),
        "history_summary": _short_text(history_summary, 2200),
        "notepad_path": _NOTEPAD_PATH,
        "notepad_exists": notepad.get("exists", False),
        "notepad": _short_text(notepad.get("content", ""), 1800),
        "context_summary": _short_text(
            state.get("context_summary", ""),
            1600,
        ),
        "compression_events": list(compression_events)[-3:],
    }

    return {
        "rules": {
            **RULES_LAYER,
            "rules": list(RULES_LAYER["rules"]),
        },
        "working_memory": working_memory,
        "history_summary_store": history_summary_store,
    }


def read_notepad(runtime: RuntimeState) -> dict[str, Any]:
    """Read ``NOTEPAD.md`` without creating it."""

    return _read_memory_file(runtime, _NOTEPAD_PATH)


def read_history_summary(runtime: RuntimeState) -> dict[str, Any]:
    """Read ``HISTORY_SUMMARY.md`` without creating it."""

    return _read_memory_file(runtime, _HISTORY_PATH)


def _read_memory_file(
    runtime: RuntimeState,
    relative_path: str,
) -> dict[str, Any]:
    target = runtime.resolve_path(relative_path)
    if not target.exists():
        return {
            "path": relative_path,
            "exists": False,
            "content": "",
        }
    if not target.is_file():
        return {
            "path": relative_path,
            "exists": False,
            "content": "",
            "error": "path is not a file",
        }
    try:
        content = target.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return {
            "path": relative_path,
            "exists": True,
            "content": "",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "path": relative_path,
        "exists": True,
        "content": content,
    }


def _short_text(text: Any, limit: int) -> str:
    """Keep at most *limit* source characters and mark truncation with ``...``."""

    if limit < 0:
        raise ValueError("limit must be greater than or equal to 0")
    if text is None:
        value = ""
    elif isinstance(text, str):
        value = text
    else:
        value = str(text)

    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _trim_handoffs(
    handoffs: Any,
) -> list[AgentHandoff]:
    """Return independent copies of the six most recent handoffs."""

    if not isinstance(handoffs, Sequence) or isinstance(
        handoffs,
        (str, bytes),
    ):
        return []
    return [
        dict(item)
        for item in list(handoffs)[-6:]
        if isinstance(item, Mapping)
    ]


def _source_prompt_items(sources: Any) -> list[SourceItem]:
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        return []

    items: list[SourceItem] = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        item: SourceItem = {
            "title": str(source.get("title") or ""),
            "url": str(source.get("url") or ""),
        }
        items.append(item)
    return items


def format_layered_memory_for_prompt(memory: LayeredMemory) -> str:
    """Serialize layered memory as readable, Unicode-preserving JSON."""

    return json.dumps(
        memory,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def memory_event(
    memory: LayeredMemory,
    *,
    node: str,
) -> dict[str, Any]:
    """Build the custom stream event emitted when a node loads memory."""

    return {
        "type": "memory",
        "node": node,
        "memory": memory,
    }
