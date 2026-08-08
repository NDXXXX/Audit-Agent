import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from audit_agent.agents.auditor import (
    AUDITOR_PROMPTS,
    _parse_findings,
    _normalize_finding,
    run_auditor,
)
from audit_agent.core.state import RuntimeState


class FakeBoundAuditorModel:
    def __init__(self, *, findings_text: str = "") -> None:
        self._findings_text = findings_text
        self.responses = iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "file_read",
                            "args": {"file_path": "src/auth.py", "offset": 0, "limit": 50},
                            "id": "call-read-1",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content=self._findings_text or _SAMPLE_FINDINGS_JSON),
            ]
        )
        self.invocations: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.invocations.append(list(messages))
        return next(self.responses)


class FakeAuditorModel:
    def __init__(self, bound: FakeBoundAuditorModel) -> None:
        self.bound = bound
        self.tools: list[StructuredTool] | None = None

    def bind_tools(self, tools: list[StructuredTool]) -> FakeBoundAuditorModel:
        self.tools = tools
        return self.bound


_SAMPLE_FINDINGS_JSON = (
    '```json\n'
    '{"findings": ['
    '{"dimension": "security", "severity": "high", "file": "src/auth.py", '
    '"line": 42, "title": "Hardcoded JWT secret", '
    '"description": "The JWT signing key is a string literal in the source code.", '
    '"suggestion": "Load from environment variable"},'
    '{"dimension": "perf", "severity": "medium", "file": "src/db.py", '
    '"line": 89, "title": "N+1 query in get_users", '
    '"description": "The function queries inside a loop.", '
    '"suggestion": "Use a JOIN or batch query"}'
    ']}\n```'
)


def test_run_auditor_parses_structured_findings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = RuntimeState.create(tmp_path / "workspace")
    state = {
        "task": "Review the auth module for security issues",
        "runtime": runtime,
        "todos": [],
    }

    bound = FakeBoundAuditorModel()
    model = FakeAuditorModel(bound)
    monkeypatch.setattr(
        "audit_agent.agents.auditor.create_model",
        lambda: model,
    )
    monkeypatch.setattr(
        "audit_agent.agents.auditor.build_layered_memory",
        lambda state, node: {"snapshot": "memory"},
    )
    # Replace build_read_only_tools to avoid needing a real workspace
    read_tool = StructuredTool.from_function(
        func=lambda file_path, offset, limit: f"content of {file_path}",
        name="file_read",
        description="Read a file.",
    )
    monkeypatch.setattr(
        "audit_agent.agents.auditor.build_read_only_tools",
        lambda runtime: [read_tool],
    )

    written_events: list[dict[str, Any]] = []
    result = run_auditor(
        state,
        "Find security issues in src/auth.py",
        dimension="security",
        writer=written_events.append,
    )

    assert result["ok"] is True
    assert result["dimension"] == "security"
    findings = result["findings"]
    assert len(findings) == 2

    sec = findings[0]
    assert sec["dimension"] == "security"
    assert sec["severity"] == "high"
    assert sec["file"] == "src/auth.py"
    assert sec["line"] == 42
    assert "JWT" in sec["title"]

    perf = findings[1]
    assert perf["dimension"] == "perf"
    assert perf["severity"] == "medium"

    # Verify system prompt was set
    first_messages = bound.invocations[0]
    assert isinstance(first_messages[0], SystemMessage)
    assert "security_auditor" in first_messages[0].content


def test_run_auditor_rejects_unknown_dimension(tmp_path: Path) -> None:
    runtime = RuntimeState.create(tmp_path / "workspace")
    with pytest.raises(ValueError, match="Unknown audit dimension"):
        run_auditor(
            {"runtime": runtime},
            "Review",
            dimension="unknown_dim",
        )


def test_run_auditor_validates_max_loops(tmp_path: Path) -> None:
    runtime = RuntimeState.create(tmp_path / "workspace")
    with pytest.raises(ValueError, match="max_loops"):
        run_auditor(
            {"runtime": runtime},
            "Review",
            dimension="security",
            max_loops=0,
        )


def test_all_auditor_prompts_are_non_empty() -> None:
    for dim in ["security", "perf", "correctness", "style"]:
        assert dim in AUDITOR_PROMPTS
        assert len(AUDITOR_PROMPTS[dim]) > 100


def test_parse_findings_extracts_json_block() -> None:
    text = (
        "Here are the issues I found:\n\n"
        '```json\n{"findings": ['
        '{"dimension": "security", "severity": "critical", '
        '"file": "x.py", "line": 1, "title": "SQLi", '
        '"description": "Unparameterized query", "suggestion": "Use params"}'
        ']}\n```'
    )
    findings = _parse_findings(text, "security")
    assert len(findings) == 1
    assert findings[0]["severity"] == "critical"


def test_parse_findings_returns_empty_on_garbage() -> None:
    assert _parse_findings("", "security") == []
    assert _parse_findings("Just some text, no JSON.", "security") == []


def test_normalize_finding_falls_back_to_medium() -> None:
    f = _normalize_finding({"title": "X"}, "security")
    assert f["severity"] == "medium"
    assert f["dimension"] == "security"


def test_normalize_finding_handles_invalid_line() -> None:
    f = _normalize_finding({"line": "not_a_number"}, "perf")
    assert f["line"] is None
