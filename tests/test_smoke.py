"""End-to-end smoke tests — full graph without monkeypatching individual nodes.

These tests verify that the graph topology, state passing, and node wiring
all work correctly.  LLM calls are satisfied by FakeModel/BoundModel instances
so no external API is consumed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from audit_agent.core.state import RuntimeState
from audit_agent.graph.workflow import (
    build_complex_workflow,
    build_entry_workflow,
)


# ---------------------------------------------------------------------------
# Fake LLM (same pattern as test_graph_nodes.py)
# ---------------------------------------------------------------------------


class FakeBoundModel:
    """Bound model that returns canned AIMessages from a shared iterator."""

    def __init__(self, responses: Any) -> None:
        # Accept any iterable (list or iterator)
        self.responses = responses if hasattr(responses, '__next__') else iter(responses)
        self.invocations: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.invocations.append(list(messages))
        return next(self.responses)


class FakeModel:
    """Model that binds tools and can be invoked."""

    def __init__(self, bound: FakeBoundModel) -> None:
        self.bound = bound
        self.bound_tools: list[BaseTool] | None = None

    def bind_tools(self, tools: list[BaseTool]) -> FakeBoundModel:
        self.bound_tools = tools
        return self.bound

    def invoke(self, messages: list[Any]) -> AIMessage:
        return self.bound.invoke(messages)


def _make_model_factory(responses: list[AIMessage]) -> Any:
    """Return a create_model replacement that shares a single response iterator."""

    shared = iter(responses)

    def factory(**kw: Any) -> FakeModel:
        return FakeModel(FakeBoundModel(shared))

    return factory


# Shared response for "passed" workflows
_PLAN = AIMessage(content=json.dumps({
    "plan_summary": "Review security of app.py",
    "todos": [{"id": "1", "content": "Security audit", "status": "pending", "note": ""}],
    "acceptance_criteria": ["No critical issues"],
    "verification_commands": [],
}))

_PASSED = AIMessage(content=json.dumps({
    "passed": True,
    "reason": "All checks passed",
    "checks": [],
    "recommended_next_instruction": "",
    "verified_findings": [],
}))

_FAILED = AIMessage(content=json.dumps({
    "passed": False,
    "reason": "Tests still fail",
    "checks": [],
    "recommended_next_instruction": "Try harder",
    "verified_findings": [],
}))


# ---------------------------------------------------------------------------
# Entry workflow smoke tests
# ---------------------------------------------------------------------------


def test_entry_workflow_routes_chat_end_to_end(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Entry graph routes chat → chat_responder → END, all real nodes wired."""

    from audit_agent.providers import deepseek_provider

    runtime = RuntimeState.create(tmp_path)

    monkeypatch.setattr(
        deepseek_provider,
        "create_model",
        _make_model_factory([
            AIMessage(content='{"route":"chat","reason":"greeting","confidence":0.99}'),
            AIMessage(content="你好！我是 Audit Agent。有什么可以帮你的？"),
        ]),
    )

    result = build_entry_workflow().invoke(
        {"task": "你好", "runtime": runtime, "messages": []}
    )

    assert result["intent_route"] == "chat"
    assert len(result.get("chat_response", "")) > 0
    assert result["chat_response"] == result.get("final_answer", "")


def test_entry_workflow_routes_workflow_end_to_end(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Entry graph routes workflow → END, skipping chat_responder."""

    from audit_agent.providers import deepseek_provider

    runtime = RuntimeState.create(tmp_path)

    monkeypatch.setattr(
        deepseek_provider,
        "create_model",
        _make_model_factory([
            AIMessage(content='{"route":"workflow","reason":"code review","confidence":0.95}'),
        ]),
    )

    result = build_entry_workflow().invoke(
        {"task": "审查 src/main.py", "runtime": runtime, "messages": []}
    )

    assert result["intent_route"] == "workflow"
    assert "chat_response" not in result


# ---------------------------------------------------------------------------
# Complex workflow smoke tests
# ---------------------------------------------------------------------------


def test_complex_workflow_success_path_end_to_end(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Planner → verifier → final, all real nodes wired, no monkeypatched nodes."""

    from audit_agent.providers import deepseek_provider
    from audit_agent.agents import auditor as auditor_mod
    from audit_agent.agents import search_agent as search_mod

    runtime = RuntimeState.create(tmp_path)
    (tmp_path / "NOTEPAD.md").write_text("", encoding="utf-8")
    (tmp_path / "HISTORY_SUMMARY.md").write_text("", encoding="utf-8")

    # Stub auditor + search agent
    monkeypatch.setattr(
        auditor_mod, "run_auditor",
        lambda state, instruction, *, dimension, writer, max_loops=5: {
            "ok": True, "dimension": dimension,
            "summary": f"{dimension}: no issues",
            "findings": [{"dimension": dimension, "severity": "low", "file": "app.py",
                          "line": 10, "title": "ok", "description": "fine", "suggestion": ""}],
            "messages": [], "tool_events": [],
        },
    )
    monkeypatch.setattr(
        search_mod, "run_search_agent",
        lambda state, instruction, *, writer, max_loops=4: {
            "ok": True, "summary": "", "sources": [], "queries": [], "messages": [],
        },
    )

    factory = _make_model_factory([_PLAN, _PASSED])
    monkeypatch.setattr(deepseek_provider, "create_model", factory)
    monkeypatch.setattr("audit_agent.graph.nodes.create_model", factory)

    result = build_complex_workflow().invoke(
        {"task": "审查 app.py", "runtime": runtime,
         "attempts": 0, "max_attempts": 2, "messages": []}
    )

    assert result["passed"] is True
    assert "Status: PASSED" in result["final_answer"]


def test_complex_workflow_failure_exhausts_attempts(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Graph retries after planner revision, fails after max_attempts exhausted."""

    from audit_agent.providers import deepseek_provider
    from audit_agent.agents import auditor as auditor_mod
    from audit_agent.agents import search_agent as search_mod

    runtime = RuntimeState.create(tmp_path)
    (tmp_path / "NOTEPAD.md").write_text("", encoding="utf-8")
    (tmp_path / "HISTORY_SUMMARY.md").write_text("", encoding="utf-8")

    monkeypatch.setattr(
        auditor_mod, "run_auditor",
        lambda state, instruction, *, dimension, writer, max_loops=5: {
            "ok": True, "dimension": dimension,
            "summary": f"{dimension}: complete",
            "findings": [], "messages": [], "tool_events": [],
        },
    )
    monkeypatch.setattr(
        search_mod, "run_search_agent",
        lambda state, instruction, *, writer, max_loops=4: {
            "ok": True, "summary": "", "sources": [], "queries": [], "messages": [],
        },
    )

    # Planner plan + verifier fail → both consumed → goes to final (attempts >= max)
    factory = _make_model_factory([_PLAN, _FAILED])
    monkeypatch.setattr(deepseek_provider, "create_model", factory)
    monkeypatch.setattr("audit_agent.graph.nodes.create_model", factory)

    result = build_complex_workflow().invoke(
        {"task": "Fix the project", "runtime": runtime,
         "attempts": 1, "max_attempts": 2, "messages": []}
    )

    assert result["passed"] is False
    assert "FAILED" in result["final_answer"]


def test_complex_workflow_compresses_context(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Verify the context compression path runs without errors."""

    from audit_agent.providers import deepseek_provider
    from audit_agent.agents import auditor as auditor_mod
    from audit_agent.agents import search_agent as search_mod

    runtime = RuntimeState.create(tmp_path)
    (tmp_path / "NOTEPAD.md").write_text("", encoding="utf-8")
    (tmp_path / "HISTORY_SUMMARY.md").write_text("", encoding="utf-8")

    monkeypatch.setattr(
        auditor_mod, "run_auditor",
        lambda state, instruction, *, dimension, writer, max_loops=5: {
            "ok": True, "dimension": dimension,
            "summary": "", "findings": [], "messages": [], "tool_events": [],
        },
    )
    monkeypatch.setattr(
        search_mod, "run_search_agent",
        lambda state, instruction, *, writer, max_loops=4: {
            "ok": True, "summary": "", "sources": [], "queries": [], "messages": [],
        },
    )

    factory = _make_model_factory([_PLAN, _PASSED])
    monkeypatch.setattr(deepseek_provider, "create_model", factory)
    monkeypatch.setattr("audit_agent.graph.nodes.create_model", factory)

    result = build_complex_workflow().invoke(
        {"task": "Simple task", "runtime": runtime,
         "attempts": 0, "max_attempts": 3, "messages": []}
    )

    assert result["passed"] is True
    assert "Status: PASSED" in result["final_answer"]
