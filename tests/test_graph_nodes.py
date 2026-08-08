import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from audit_agent.core.approval import ApprovalDecision, ApprovalRequest
from audit_agent.core.state import RuntimeState
from audit_agent.graph import nodes
from audit_agent.graph.state import AuditGraphState


class FakeBoundModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = iter(responses)
        self.invocations: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.invocations.append(list(messages))
        return next(self.responses)


class FakeModel:
    def __init__(self, bound: FakeBoundModel) -> None:
        self.bound = bound
        self.bound_tools: list[Any] | None = None

    def bind_tools(self, tools: list[Any]) -> FakeBoundModel:
        self.bound_tools = tools
        return self.bound

    def invoke(self, messages: list[Any]) -> AIMessage:
        return self.bound.invoke(messages)


class FakeTokenCountingModel:
    def __init__(self, token_count: int | Exception) -> None:
        self.token_count = token_count
        self.messages: list[Any] = []

    def get_num_tokens_from_messages(self, messages: list[Any]) -> int:
        self.messages = list(messages)
        if isinstance(self.token_count, Exception):
            raise self.token_count
        return self.token_count


class FakeCompressionModel:
    def __init__(self, response: AIMessage, token_count: int) -> None:
        self.response = response
        self.token_count = token_count
        self.invocations: list[list[Any]] = []
        self.counted_messages: list[Any] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.invocations.append(list(messages))
        return self.response

    def get_num_tokens_from_messages(self, messages: list[Any]) -> int:
        self.counted_messages = list(messages)
        return self.token_count


def _todo(
    *,
    status: str = "pending",
    note: str = "",
) -> dict[str, str]:
    return {
        "id": "todo-1",
        "content": "Create the requested file",
        "status": status,
        "note": note,
    }


def test_intent_router_selects_chat_and_injects_session_memory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = FakeBoundModel(
        [
            AIMessage(
                content=json.dumps(
                    {
                        "route": "chat",
                        "reason": "This is a greeting.",
                        "confidence": 0.94,
                    }
                )
            )
        ]
    )
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(nodes, "create_model", lambda: model)
    monkeypatch.setattr(nodes, "_get_event_writer", lambda: events.append)

    update = nodes.intent_router_node(
        {
            "task": "你好",
            "runtime": RuntimeState(tmp_path),
            "context_summary": "A previous coding task is complete.",
        }
    )

    assert update == {
        "intent_route": "chat",
        "intent_reason": "This is a greeting.",
        "intent_confidence": 0.94,
    }
    assert events[0]["type"] == "memory"
    assert events[0]["node"] == "intent_router"
    assert model.invocations[0][0].content == nodes.INTENT_ROUTER_PROMPT
    request_text, memory_text = model.invocations[0][1].content.split(
        "\n\nLayered memory:\n",
        1,
    )
    request = json.loads(request_text)
    assert request["latest_input"] == "你好"
    assert request["workflow_context"]["context_summary"] == (
        "A previous coding task is complete."
    )
    assert json.loads(memory_text)["working_memory"]["node"] == (
        "intent_router"
    )


@pytest.mark.parametrize(
    "response_content",
    [
        json.dumps(
            {
                "route": "chat",
                "reason": "Uncertain greeting.",
                "confidence": 0.54,
            }
        ),
        json.dumps(
            {
                "route": "unsupported",
                "reason": "Invalid route.",
                "confidence": 0.9,
            }
        ),
        "not JSON",
    ],
)
def test_intent_router_defaults_uncertain_or_invalid_output_to_workflow(
    response_content: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        nodes,
        "create_model",
        lambda: FakeBoundModel([AIMessage(content=response_content)]),
    )

    update = nodes.intent_router_node(
        {
            "task": "继续",
            "runtime": RuntimeState(tmp_path),
        }
    )

    assert update["intent_route"] == "workflow"
    assert 0.0 <= update["intent_confidence"] <= 1.0


def test_chat_responder_answers_without_binding_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = FakeBoundModel([AIMessage(content="你好！我是 Audit Agent。")])
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(nodes, "create_model", lambda: model)
    monkeypatch.setattr(nodes, "_get_event_writer", lambda: events.append)

    update = nodes.chat_responder_node(
        {
            "task": "你是谁？",
            "runtime": RuntimeState(tmp_path),
        }
    )

    assert update == {
        "chat_response": "你好！我是 Audit Agent。",
        "final_answer": "你好！我是 Audit Agent。",
    }
    assert model.invocations[0][0].content == nodes.CHAT_RESPONDER_PROMPT
    assert events[0]["node"] == "chat_responder"


@pytest.mark.parametrize(
    ("intent_route", "expected"),
    [
        ("chat", "chat_responder"),
        ("workflow", "planner"),
        ("invalid", "planner"),
        (None, "planner"),
    ],
)
def test_intent_route_fn(
    intent_route: str | None,
    expected: str,
) -> None:
    state: AuditGraphState = {}
    if intent_route is not None:
        state["intent_route"] = intent_route
    assert nodes.intent_route_fn(state) == expected


def test_planner_node_creates_structured_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "todo_write",
                "args": {
                    "plan_summary": "Create and verify one file.",
                    "todos": [_todo()],
                    "acceptance_criteria": ["The file exists."],
                    "verification_commands": ["test -f result.txt"],
                },
                "id": "plan-call",
                "type": "tool_call",
            }
        ],
    )
    bound = FakeBoundModel(
        [
            plan_response,
            AIMessage(content="Plan published; no delegation was needed."),
        ]
    )
    model = FakeModel(bound)
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(nodes, "create_model", lambda: model)
    monkeypatch.setattr(nodes, "_get_event_writer", lambda: events.append)

    update = nodes.planner_node(
        {
            "task": "Create result.txt",
            "runtime": RuntimeState(tmp_path),
        }
    )

    assert update["plan_summary"] == "Create and verify one file."
    assert update["todos"] == [_todo()]
    assert update["acceptance_criteria"] == ["The file exists."]
    assert update["verification_commands"] == ["test -f result.txt"]
    assert update["context_next_node"] == "verifier"
    assert model.bound_tools is not None
    assert [tool.name for tool in model.bound_tools] == [
        "todo_write",
        "call_search_agent",
        "call_auditors",
    ]
    assert [event["type"] for event in events] == [
        "memory",
        "tool_call",
        "tool_result",
    ]
    request_text, memory_text = bound.invocations[0][1].content.split(
        "\n\nLayered memory:\n",
        1,
    )
    assert json.loads(request_text)["task"] == "Create result.txt"
    assert json.loads(memory_text)["working_memory"]["node"] == "planner"


def test_planner_node_revises_failed_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "todo_write",
                "args": {
                    "plan_summary": "Revise the failed implementation.",
                    "todos": [_todo(note="Revised")],
                    "acceptance_criteria": ["Tests pass."],
                    "verification_commands": ["python -m pytest"],
                },
                "id": "revised-plan",
                "type": "tool_call",
            }
        ],
    )
    bound = FakeBoundModel(
        [response, AIMessage(content="Revised the failed plan.")]
    )
    monkeypatch.setattr(nodes, "create_model", lambda: FakeModel(bound))

    update = nodes.planner_node(
        {
            "task": "Fix the project",
            "runtime": RuntimeState(tmp_path),
            "todos": [_todo(status="blocked")],
            "attempts": 1,
            "passed": False,
            "last_error": "Tests failed",
        }
    )

    assert update["plan_summary"] == "Revise the failed implementation."
    request_text, memory_text = bound.invocations[0][1].content.split(
        "\n\nLayered memory:\n",
        1,
    )
    request = json.loads(request_text)
    assert request["mode"] == "revise"
    assert request["last_error"] == "Tests failed"
    assert json.loads(memory_text)["working_memory"]["last_error"] == (
        "Tests failed"
    )


def test_planner_node_delegates_search_then_code_and_persists_handoffs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    planned_todo = _todo()
    first = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "todo_write",
                "args": {
                    "plan_summary": "Research, implement, and verify.",
                    "todos": [planned_todo],
                    "acceptance_criteria": ["The result is sourced."],
                    "verification_commands": ["python -m pytest"],
                },
                "id": "plan-call",
                "type": "tool_call",
            },
            {
                "name": "call_search_agent",
                "args": {"instruction": "Find the official specification."},
                "id": "search-handoff",
                "type": "tool_call",
            },
            {
                "name": "call_auditors",
                "args": {"instruction": "Implement the published plan."},
                "id": "audit-handoff",
                "type": "tool_call",
            },
        ],
    )
    bound = FakeBoundModel(
        [first, AIMessage(content="Specialists completed the planned work.")]
    )
    model = FakeModel(bound)
    events: list[dict[str, Any]] = []
    code_state_snapshots: list[dict[str, Any]] = []

    def fake_search_agent(
        state: dict[str, Any],
        instruction: str,
        *,
        writer: Any,
    ) -> dict[str, Any]:
        assert state["plan_summary"] == "Research, implement, and verify."
        assert instruction == "Find the official specification."
        return {
            "ok": True,
            "summary": "Official specification located.",
            "sources": ["https://example.com/spec"],
            "messages": [],
            "tool_events": [
                {
                    "type": "search_results",
                    "results": [
                        {
                            "title": "Official specification",
                            "url": "https://example.com/spec",
                            "content": "Specification text",
                            "score": 0.99,
                        }
                    ],
                }
            ],
        }

    def fake_auditor(
        state: dict[str, Any],
        instruction: str,
        *,
        dimension: str,
        writer: Any,
    ) -> dict[str, Any]:
        code_state_snapshots.append(dict(state))
        return {
            "ok": True,
            "dimension": dimension,
            "summary": f"Audit {dimension} completed.",
            "findings": [
                {
                    "dimension": dimension,
                    "severity": "medium",
                    "file": "test.py",
                    "line": 1,
                    "title": f"Finding from {dimension}",
                    "description": "Test",
                    "suggestion": "Fix it",
                }
            ],
            "messages": [],
            "tool_events": [],
        }

    monkeypatch.setattr(nodes, "create_model", lambda: model)
    monkeypatch.setattr(nodes, "run_search_agent", fake_search_agent)
    monkeypatch.setattr(nodes, "run_auditor", fake_auditor)
    monkeypatch.setattr(nodes, "_get_event_writer", lambda: events.append)

    state = {
        "task": "Build researched content",
        "runtime": RuntimeState(tmp_path),
        "todos": [],
    }

    update = nodes.planner_node(state)

    assert update["todos"] == [planned_todo]
    assert update["research_notes"] == "Official specification located."
    assert update["sources"] == [
        {
            "title": "Official specification",
            "url": "https://example.com/spec",
            "content": "Specification text",
            "score": 0.99,
        }
    ]
    # Auditors return review_findings, not code_agent_summary
    findings = update["review_findings"]
    assert len(findings) == 4  # one per audit dimension
    assert all(
        f["dimension"] in {"security", "perf", "correctness", "style"}
        for f in findings
    )
    assert update["agent_handoffs"][0] == {
        "from_agent": "planner",
        "to_agent": "searchAgent",
        "instruction": "Find the official specification.",
        "result": "Official specification located.",
    }
    # Second handoff is the combined auditors handoff
    assert update["agent_handoffs"][1]["to_agent"] == "auditors"
    assert "messages" not in update  # auditors don't return messages in test
    assert code_state_snapshots[0]["research_notes"] == (
        "Official specification located."
    )
    assert events[0]["type"] == "memory"
    assert events[0]["node"] == "planner"
    assert [
        event
        for event in events
        if event.get("type") == "handoff"
    ] == [
        {
            "type": "handoff",
            "from": "planner",
            "to": "searchAgent",
            "instruction": "Find the official specification.",
        },
        {
            "type": "handoff",
            "from": "planner",
            "to": "auditor:security",
            "instruction": "Implement the published plan.",
        },
        {
            "type": "handoff",
            "from": "planner",
            "to": "auditor:perf",
            "instruction": "Implement the published plan.",
        },
        {
            "type": "handoff",
            "from": "planner",
            "to": "auditor:correctness",
            "instruction": "Implement the published plan.",
        },
        {
            "type": "handoff",
            "from": "planner",
            "to": "auditor:style",
            "instruction": "Implement the published plan.",
        },
    ]


def test_verifier_node_passes_and_completes_todos(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    response = AIMessage(
        content=json.dumps(
            {
                "passed": True,
                "reason": "All acceptance criteria are satisfied.",
                "checks": [
                    {
                        "name": "file exists",
                        "passed": True,
                        "detail": "result.txt was inspected",
                    }
                ],
                "recommended_next_instruction": "",
            }
        )
    )
    model = FakeModel(FakeBoundModel([response]))
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(nodes, "create_model", lambda: model)
    monkeypatch.setattr(nodes, "build_read_only_tools", lambda runtime: [])
    monkeypatch.setattr(nodes, "_get_event_writer", lambda: events.append)
    monkeypatch.setattr(
        nodes,
        "_run_verification_command",
        lambda command, runtime: {
            "command": command,
            "ok": True,
            "exit_code": 0,
            "stdout": "ok",
            "stderr": "",
        },
    )

    update = nodes.verifier_node(
        {
            "task": "Create result.txt",
            "runtime": RuntimeState(tmp_path),
            "todos": [_todo(status="in_progress")],
            "acceptance_criteria": ["The file exists."],
            "verification_commands": ["test -f result.txt"],
            "attempts": 1,
        }
    )

    assert update["passed"] is True
    assert update["attempts"] == 2
    assert update["last_error"] == ""
    assert update["todos"][0]["status"] == "completed"
    assert update["context_next_node"] == "final"
    assert len(update["verification_results"]) == 1
    assert len(update["verification_checks"]) == 2
    assert model.bound_tools == []
    assert events[0]["type"] == "memory"
    assert events[0]["node"] == "verifier"
    request_text, memory_text = model.bound.invocations[0][1].content.split(
        "\n\nLayered memory:\n",
        1,
    )
    assert json.loads(request_text)["verification_results"][0]["ok"] is True
    assert json.loads(memory_text)["working_memory"]["node"] == "verifier"


def test_verifier_node_command_failure_overrides_model_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    response = AIMessage(
        content=json.dumps(
            {
                "passed": True,
                "reason": "The source looks correct.",
                "checks": [],
                "recommended_next_instruction": "",
            }
        )
    )
    monkeypatch.setattr(
        nodes,
        "create_model",
        lambda: FakeModel(FakeBoundModel([response])),
    )
    monkeypatch.setattr(nodes, "build_read_only_tools", lambda runtime: [])
    monkeypatch.setattr(
        nodes,
        "_run_verification_command",
        lambda command, runtime: {
            "command": command,
            "ok": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": "failed",
        },
    )

    update = nodes.verifier_node(
        {
            "runtime": RuntimeState(tmp_path),
            "todos": [_todo(status="in_progress")],
            "verification_commands": ["false"],
        }
    )

    assert update["passed"] is False
    assert "Failed verification command" in update["last_error"]
    assert update["todos"][0]["status"] == "blocked"
    assert update["context_next_node"] == "planner"


def test_verifier_forces_final_json_after_read_only_tool_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool_responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "file_read",
                    "args": {"file_path": "result.txt"},
                    "id": f"read-{index}",
                }
            ],
        )
        for index in range(6)
    ]
    final_response = AIMessage(
        content=json.dumps(
            {
                "passed": True,
                "reason": "The deterministic checks and inspected file pass.",
                "checks": [],
                "recommended_next_instruction": "",
            }
        )
    )
    model = FakeModel(FakeBoundModel([*tool_responses, final_response]))
    monkeypatch.setattr(nodes, "create_model", lambda: model)
    monkeypatch.setattr(nodes, "build_read_only_tools", lambda runtime: [])
    monkeypatch.setattr(
        nodes,
        "execute_tool",
        lambda call, *, tools_by_name: {"ok": True, "content": "inspected"},
    )
    monkeypatch.setattr(
        nodes,
        "_run_verification_command",
        lambda command, runtime: {
            "command": command,
            "ok": True,
            "exit_code": 0,
            "stdout": "passed",
            "stderr": "",
        },
    )

    update = nodes.verifier_node(
        {
            "runtime": RuntimeState(tmp_path),
            "todos": [_todo(status="completed")],
            "verification_commands": ["pytest -q"],
        }
    )

    assert update["passed"] is True
    assert len(model.bound.invocations) == 7
    forced_prompt = model.bound.invocations[-1][-1]
    assert isinstance(forced_prompt, HumanMessage)
    assert "Do not call any more tools" in forced_prompt.content


def test_context_monitor_node_uses_model_token_counter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = FakeTokenCountingModel(401_000)
    monkeypatch.setattr(nodes, "create_model", lambda: model)

    update = nodes.context_monitor_node(
        {
            "task": "Create result.txt",
            "runtime": RuntimeState(tmp_path),
            "messages": [HumanMessage(content="Implement the task")],
            "context_token_limit": 400_000,
            "context_next_node": "planner",
        }
    )

    assert update == {
        "context_token_count": 401_000,
        "context_should_compress": True,
        "context_next_node": "planner",
    }
    assert len(model.messages) == 2
    assert isinstance(model.messages[-1], HumanMessage)
    memory_payload = json.loads(model.messages[-1].content)
    assert memory_payload["working_memory"]["node"] == "context_monitor"
    assert memory_payload["working_memory"]["task"] == "Create result.txt"


def test_context_monitor_node_falls_back_to_character_estimate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        nodes,
        "create_model",
        lambda: (_ for _ in ()).throw(RuntimeError("tokenizer unavailable")),
    )
    state: AuditGraphState = {
        "task": "Fallback counting",
        "runtime": RuntimeState(tmp_path),
        "messages": [HumanMessage(content="abcd")],
    }
    memory = nodes.build_layered_memory(state, node="context_monitor")
    memory_text = nodes.format_layered_memory_for_prompt(memory)

    update = nodes.context_monitor_node(state)

    assert update == {
        "context_token_count": len(f"abcd\n{memory_text}") // 4,
        "context_should_compress": False,
        "context_next_node": "verifier",
    }


def test_context_compressor_node_replaces_messages_and_persists_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compression_payload = {
        "summary": "Continue by fixing the remaining parser test.",
        "active_goal": "Make all tests pass",
        "completed_work": ["Implemented the parser"],
        "open_todos": ["Fix the edge case"],
        "important_files": ["parser.py", "test_parser.py"],
        "tool_findings": ["One test still fails"],
        "sources": ["https://example.com/spec"],
        "next_steps": ["Patch and rerun tests"],
        "risks": ["Preserve the public API"],
    }
    model = FakeCompressionModel(
        AIMessage(content=json.dumps(compression_payload)),
        token_count=23,
    )
    monkeypatch.setattr(nodes, "create_model", lambda: model)
    previous_event = {
        "node": "context_compressor",
        "token_count_before": 500_000,
        "token_count_after": 100_000,
    }

    update = nodes.context_compressor_node(
        {
            "task": "Fix the parser",
            "runtime": RuntimeState(tmp_path),
            "messages": [
                HumanMessage(content="Implement the parser"),
                AIMessage(content="I created parser.py"),
            ],
            "plan_summary": "p" * 2_000,
            "todos": [_todo(note="n" * 900)],
            "acceptance_criteria": ["a" * 900],
            "verification_commands": ["c" * 1_100],
            "research_notes": "r" * 1_700,
            "sources": [
                {
                    "title": "Official spec",
                    "url": "https://example.com/spec",
                    "content": "s" * 700,
                    "score": 0.9,
                }
            ],
            "agent_handoffs": [
                {
                    "from_agent": "planner",
                    "to_agent": "codeAgent",
                    "instruction": str(index),
                    "result": "x" * 1_100,
                }
                for index in range(8)
            ],
            "verification_results": [
                {
                    "command": "python -m pytest",
                    "ok": False,
                    "exit_code": 1,
                    "stdout": "o" * 2_100,
                    "stderr": "s" * 2_100,
                }
            ],
            "verification_checks": [
                {
                    "name": "tests",
                    "passed": False,
                    "detail": "d" * 1_100,
                }
            ],
            "verification_reason": "v" * 1_500,
            "recommended_next_instruction": "i" * 1_500,
            "last_error": "e" * 1_500,
            "context_token_count": 450_000,
            "context_should_compress": True,
            "session_turn": 4,
            "compression_events": [previous_event],
        }
    )

    assert isinstance(update["messages"][0], RemoveMessage)
    assert update["messages"][0].id == REMOVE_ALL_MESSAGES
    assert isinstance(update["messages"][1], AIMessage)
    assert update["messages"][1].content == compression_payload["summary"]
    assert update["context_summary"] == compression_payload["summary"]
    assert update["history_summary"] == compression_payload["summary"]
    assert update["context_token_count"] == 23
    assert update["context_should_compress"] is False
    assert (tmp_path / "HISTORY_SUMMARY.md").read_text(
        encoding="utf-8"
    ) == compression_payload["summary"]

    request = json.loads(model.invocations[0][1].content)
    assert len(request["messages"]) == 2
    assert request["layered_memory"]["working_memory"]["node"] == (
        "context_compressor"
    )
    assert model.counted_messages[0].content == compression_payload["summary"]

    assert len(update["plan_summary"]) == 1_603
    assert len(update["todos"][0]["note"]) == 803
    assert len(update["acceptance_criteria"][0]) == 803
    assert len(update["verification_commands"][0]) == 1_003
    assert len(update["research_notes"]) == 1_603
    assert len(update["sources"][0]["content"]) == 603
    assert len(update["agent_handoffs"]) == 6
    assert update["agent_handoffs"][0]["instruction"] == "2"
    assert len(update["agent_handoffs"][0]["result"]) == 1_003
    assert len(update["verification_results"][0]["stdout"]) == 2_003
    assert len(update["verification_results"][0]["stderr"]) == 2_003
    assert len(update["verification_checks"][0]["detail"]) == 1_003
    assert len(update["verification_reason"]) == 1_403
    assert len(update["recommended_next_instruction"]) == 1_403
    assert len(update["last_error"]) == 1_403

    assert update["compression_events"][0] == previous_event
    new_event = update["compression_events"][-1]
    assert new_event == {
        "node": "context_compressor",
        "reason": "Context exceeded the configured token limit.",
        "session_turn": 4,
        "token_count_before": 450_000,
        "token_count_after": 23,
        "summary": compression_payload["summary"],
    }


def test_context_compressor_replaces_history_through_graph_reducer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    summary = "Only the compressed context remains."
    model = FakeCompressionModel(
        AIMessage(
            content=json.dumps(
                {
                    "summary": summary,
                    "active_goal": "Continue",
                    "completed_work": [],
                    "open_todos": [],
                    "important_files": [],
                    "tool_findings": [],
                    "sources": [],
                    "next_steps": [],
                    "risks": [],
                }
            )
        ),
        token_count=8,
    )
    monkeypatch.setattr(nodes, "create_model", lambda: model)

    builder = StateGraph(AuditGraphState)
    builder.add_node("context_compressor", nodes.context_compressor_node)
    builder.add_edge(START, "context_compressor")
    builder.add_edge("context_compressor", END)

    result = builder.compile().invoke(
        {
            "runtime": RuntimeState(tmp_path),
            "messages": [
                HumanMessage(content="old user message"),
                AIMessage(content="old assistant message"),
            ],
            "context_should_compress": True,
        }
    )

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == summary


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {
                "passed": True,
                "context_should_compress": True,
                "context_next_node": "planner",
            },
            "final",
        ),
        (
            {
                "passed": False,
                "context_should_compress": True,
                "context_next_node": "verifier",
            },
            "context_compressor",
        ),
        (
            {
                "passed": False,
                "context_should_compress": False,
                "context_next_node": "planner",
            },
            "planner",
        ),
        ({"passed": False}, "verifier"),
    ],
)
def test_context_monitor_route(
    state: AuditGraphState,
    expected: str,
) -> None:
    assert nodes.context_monitor_route(state) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"context_next_node": "planner"}, "planner"),
        ({"context_next_node": "final"}, "final"),
        ({}, "verifier"),
    ],
)
def test_context_compressor_route(
    state: AuditGraphState,
    expected: str,
) -> None:
    assert nodes.context_compressor_route(state) == expected


def test_verification_command_captures_process_result(tmp_path: Path) -> None:
    result = nodes._run_verification_command(
        "printf verified",
        RuntimeState(tmp_path),
    )

    assert result == {
        "command": "printf verified",
        "ok": True,
        "exit_code": 0,
        "stdout": "verified",
        "stderr": "",
    }


def test_verifier_reuses_same_attempt_rejected_dependency_approval(
    tmp_path: Path,
) -> None:
    requests: list[ApprovalRequest] = []

    def reject(request: ApprovalRequest) -> ApprovalDecision:
        requests.append(request)
        return ApprovalDecision(False, "Rejected for this attempt.")

    runtime = RuntimeState(tmp_path, approval_handler=reject)
    first = nodes._run_verification_command(
        "uv sync --project recovery_lab 2>&1; echo complete",
        runtime,
    )
    second = nodes._run_verification_command(
        "uv sync --project recovery_lab",
        runtime,
    )

    assert len(requests) == 1
    assert first["ok"] is False
    assert second["ok"] is False
    assert "Rejected for this attempt" in first["stderr"]
    assert "Rejected for this attempt" in second["stderr"]


def test_failed_verification_reopens_a_completed_verification_todo() -> None:
    todos = [
        _todo(status="completed", note="Implementation done"),
        {
            "id": "todo-verify",
            "content": "Run tests and verify output",
            "status": "completed",
            "note": "Agent claimed success",
        },
    ]

    updated = nodes._verified_todos(
        todos,
        passed=False,
        failure_note="Dependency synchronization was rejected.",
    )

    assert updated[0]["status"] == "completed"
    assert updated[1]["status"] == "blocked"
    assert updated[1]["note"] == "Dependency synchronization was rejected."


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"passed": True, "attempts": 1, "max_attempts": 3}, "final"),
        ({"passed": False, "attempts": 3, "max_attempts": 3}, "final"),
        ({"passed": False, "attempts": 1, "max_attempts": 3}, "planner"),
    ],
)
def test_verifier_route(
    state: dict[str, Any],
    expected: str,
) -> None:
    assert nodes.verifier_route(state) == expected
