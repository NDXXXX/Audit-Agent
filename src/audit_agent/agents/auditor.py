"""Focused ReAct agent for code review across security, perf, correctness, and style dimensions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from audit_agent.core.state import RuntimeState
from audit_agent.graph.memory import (
    build_layered_memory,
    format_layered_memory_for_prompt,
    memory_event,
)
from audit_agent.providers import create_model
from audit_agent.tools.execution import (
    EventWriter,
    _record_event,
    execute_tool,
)
from audit_agent.tools.registry import build_read_only_tools
from audit_agent.tools.web_search_tool import WebSearchTool

AUDITOR_PROMPTS = {
    "security": (
        "You are security_auditor, a focused security reviewer.\n\n"
        "Find security vulnerabilities in the code under review. Look for:\n"
        "- Injection vulnerabilities (SQL, command, template, log)\n"
        "- Authentication and authorization flaws\n"
        "- Hardcoded secrets, keys, or tokens\n"
        "- Unsafe deserialization or eval usage\n"
        "- Path traversal and file access issues\n"
        "- Missing or incorrect input validation\n"
        "- Insecure cryptography (weak ciphers, broken hashes, missing salts)\n"
        "- CORS/CSRF/XSS issues in web code\n\n"
        "Rules:\n"
        "- Use file_read to inspect suspicious files.\n"
        "- Use grep to search for risky patterns across the codebase.\n"
        "- Use WebSearchTool to check CVEs or vulnerability databases when relevant.\n"
        "- You must NOT modify any files.\n"
        "- Return findings as structured JSON at the end of your response.\n"
    ),
    "perf": (
        "You are perf_auditor, a focused performance reviewer.\n\n"
        "Find performance issues in the code under review. Look for:\n"
        "- N+1 query patterns and missing database indexes\n"
        "- O(n²) or worse algorithmic complexity\n"
        "- Unbounded memory allocations or leaks\n"
        "- Blocking I/O in async or hot paths\n"
        "- Missing caching opportunities\n"
        "- Inefficient data structures (e.g., list for membership checks)\n"
        "- Excessive object creation in loops\n"
        "- Missing lazy evaluation or pagination\n\n"
        "Rules:\n"
        "- Use file_read to inspect suspicious files.\n"
        "- Use grep to find patterns like nested loops, repeated queries.\n"
        "- You must NOT modify any files.\n"
        "- Return findings as structured JSON at the end of your response.\n"
    ),
    "correctness": (
        "You are correctness_auditor, a focused logic and correctness reviewer.\n\n"
        "Find logic bugs and correctness issues in the code under review. Look for:\n"
        "- Off-by-one errors and boundary condition mistakes\n"
        "- Null/None reference risks and missing null checks\n"
        "- Race conditions and concurrency bugs\n"
        "- Inverted or incorrect boolean conditions\n"
        "- Missing error handling or swallowed exceptions\n"
        "- Type confusion and incorrect type assumptions\n"
        "- Broken edge cases (empty input, max values, negative numbers)\n"
        "- Incorrect state management (stale state, missing cleanup)\n\n"
        "Rules:\n"
        "- Use file_read to inspect suspicious files.\n"
        "- Use grep to trace control flow across files.\n"
        "- You must NOT modify any files.\n"
        "- Return findings as structured JSON at the end of your response.\n"
    ),
    "style": (
        "You are style_auditor, a focused code style and maintainability reviewer.\n\n"
        "Find style and maintainability issues in the code under review. Look for:\n"
        "- Naming convention violations (functions, variables, classes)\n"
        "- Missing or incorrect docstrings and comments\n"
        "- Overly complex functions (high cyclomatic complexity)\n"
        "- Dead code, unused imports, and unreachable branches\n"
        "- Inconsistent formatting or code organization\n"
        "- Missing type annotations where they would add clarity\n"
        "- Functions that do too many things (single responsibility violations)\n"
        "- Magic numbers and hardcoded values without explanation\n\n"
        "Rules:\n"
        "- Use file_read to inspect suspicious files.\n"
        "- Use grep to find patterns across the codebase.\n"
        "- You must NOT modify any files.\n"
        "- Return findings as structured JSON at the end of your response.\n"
    ),
}

_FINDING_SCHEMA_HINT = (
    "\n\nAfter your analysis, you MUST output a JSON object with a "
    '"findings" key containing a list of findings. Each finding must have: '
    "dimension (string), severity (one of: critical, high, medium, low), "
    "file (string, relative path), line (integer or null), "
    'title (string, one-line summary), description (string, detailed explanation), '
    "suggestion (string, how to fix). "
    "If you found no issues, return an empty findings list.\n"
    'Example: {"findings": [{"dimension": "security", "severity": "high", '
    '"file": "src/auth.py", "line": 42, "title": "Hardcoded JWT secret", '
    '"description": "The JWT signing key is a string literal...", '
    '"suggestion": "Load the secret from an environment variable"}]}'
)

def run_auditor(
    state: Mapping[str, Any],
    instruction: str,
    *,
    dimension: str,
    writer: EventWriter | None = None,
    max_loops: int = 5,
) -> dict[str, Any]:
    """Run one dimension-specific code review and return structured findings."""

    if max_loops < 1:
        raise ValueError("max_loops must be greater than or equal to 1")
    if dimension not in AUDITOR_PROMPTS:
        raise ValueError(
            f"Unknown audit dimension {dimension!r}. "
            f"Choose from: {', '.join(AUDITOR_PROMPTS)}"
        )

    runtime = _require_runtime(state)
    web_search = WebSearchTool()
    search_tool = web_search.as_structured_tool()
    tools = [*build_read_only_tools(runtime), search_tool]
    tools_by_name = {tool.name: tool for tool in tools}
    agent = create_model().bind_tools(tools)

    memory = build_layered_memory(state, node=f"auditor:{dimension}")
    if writer is not None:
        writer(memory_event(memory, node=f"auditor:{dimension}"))

    system_prompt = AUDITOR_PROMPTS[dimension] + _FINDING_SCHEMA_HINT
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=_auditor_input(state, instruction, memory)),
    ]
    tool_events: list[dict[str, Any]] = []
    summary = ""
    findings: list[dict[str, Any]] = []

    for _ in range(max_loops):
        response = agent.invoke(messages)
        messages.append(response)
        response_text = str(response.content) if response.content else ""
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
                    "agent": f"auditor:{dimension}",
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
                    "agent": f"auditor:{dimension}",
                    "name": name,
                    "result": result,
                },
                writer=writer,
                events=tool_events,
            )

    # Parse structured findings from the final summary
    findings = _parse_findings(summary, dimension)

    ok = bool(summary and (findings or "no issues" in summary.lower()))
    if not summary:
        summary = (
            f"auditor:{dimension} stopped after {max_loops} loops "
            "without a final response."
        )

    return {
        "ok": ok,
        "dimension": dimension,
        "summary": summary,
        "findings": findings,
        "messages": messages,
        "tool_events": tool_events,
    }


def _auditor_input(
    state: Mapping[str, Any],
    instruction: str,
    memory: Mapping[str, Any],
) -> str:
    """Format the review request with task context and memory snapshot."""

    request = {
        "task": state.get("task", ""),
        "instruction": instruction,
    }
    return (
        f"{json.dumps(request, ensure_ascii=False, default=str)}"
        "\n\nLayered memory:\n"
        f"{format_layered_memory_for_prompt(memory)}"
    )


def _parse_findings(
    text: str,
    dimension: str,
) -> list[dict[str, Any]]:
    """Extract structured findings from the auditor's final response."""

    if not text:
        return []

    # Try to find a JSON block in the response
    candidates: list[str] = []

    # Look for ```json ... ``` blocks
    json_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
    candidates.extend(json_blocks)

    # Also try the whole text as a fallback — the model might return raw JSON
    candidates.append(text)

    for candidate in candidates:
        try:
            start = candidate.index("{")
            end = candidate.rindex("}") + 1
            parsed = json.loads(candidate[start:end])
        except (ValueError, json.JSONDecodeError):
            continue

        if isinstance(parsed, dict):
            raw_findings = parsed.get("findings")
            if isinstance(raw_findings, list):
                return [
                    _normalize_finding(f, dimension) for f in raw_findings
                    if isinstance(f, Mapping)
                ]
            # Sometimes the model returns a single finding, not wrapped
            file = parsed.get("file")
            title = parsed.get("title")
            if file or title:
                return [_normalize_finding(parsed, dimension)]

    return []


_VALID_SEVERITIES = frozenset({"critical", "high", "medium", "low"})


def _normalize_finding(
    raw: Mapping[str, Any],
    dimension: str,
) -> dict[str, Any]:
    """Coerce a raw finding dict into the canonical ReviewFinding shape."""

    severity = str(raw.get("severity", "medium")).lower().strip()
    if severity not in _VALID_SEVERITIES:
        severity = "medium"

    line = raw.get("line")
    if line is not None:
        try:
            line = int(line)
        except (TypeError, ValueError):
            line = None

    return {
        "dimension": str(raw.get("dimension") or dimension),
        "severity": severity,
        "file": str(raw.get("file") or ""),
        "line": line,
        "title": str(raw.get("title") or ""),
        "description": str(raw.get("description") or ""),
        "suggestion": str(raw.get("suggestion") or ""),
    }


def _require_runtime(state: Mapping[str, Any]) -> RuntimeState:
    runtime = state.get("runtime")
    if not isinstance(runtime, RuntimeState):
        raise TypeError("state.runtime must be a RuntimeState")
    return runtime


