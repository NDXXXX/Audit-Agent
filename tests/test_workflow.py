from typing import Any

import pytest

from audit_agent.graph import workflow
from audit_agent.prompts.stage2 import FINAL_PROMPT
from audit_agent.prompts.stage3 import PLANNER_PROMPT, VERIFIER_PROMPT
from audit_agent.prompts.stage4 import CONTEXT_COMPRESSION_PROMPT
from audit_agent.prompts.stage5 import CHAT_RESPONDER_PROMPT, INTENT_ROUTER_PROMPT


def test_stage3_planner_and_shared_prompts_define_each_role() -> None:
    assert "planner/supervisor" in PLANNER_PROMPT
    assert "TodoWriteTool" in PLANNER_PROMPT
    assert "CallSearchAgentTool" in PLANNER_PROMPT
    assert "CallAuditorsTool" in PLANNER_PROMPT
    assert "adversarial" in VERIFIER_PROMPT.lower()
    assert "deduplicate" in VERIFIER_PROMPT.lower()
    assert "passed: boolean" in VERIFIER_PROMPT
    assert "{status}" in FINAL_PROMPT
    assert "context_compressor" in CONTEXT_COMPRESSION_PROMPT
    assert "Return only JSON" in CONTEXT_COMPRESSION_PROMPT
    assert "important_files" in CONTEXT_COMPRESSION_PROMPT
    assert '"route":"chat"|"workflow"' in INTENT_ROUTER_PROMPT
    assert "If uncertain, choose workflow" in INTENT_ROUTER_PROMPT
    assert "Do not claim that you read files" in CHAT_RESPONDER_PROMPT


@pytest.mark.parametrize(
    ("passed", "expected_status"),
    [
        (True, "Status: PASSED"),
        (False, "Status: FAILED"),
    ],
)
def test_final_node_formats_terminal_state(
    passed: bool,
    expected_status: str,
) -> None:
    update = workflow.final_node(
        {
            "task": "Create result.txt",
            "passed": passed,
            "attempts": 2,
            "max_attempts": 3,
            "verification_reason": "Verification result.",
            "recommended_next_instruction": "Fix the failed check.",
        }
    )

    assert expected_status in update["final_answer"]
    assert "Review cycles: 2/3" in update["final_answer"]


def test_build_workflow_contains_expected_nodes() -> None:
    graph = workflow.build_workflow()

    assert set(graph.get_graph().nodes) == {
        "__start__",
        "planner",
        "context_monitor",
        "context_compressor",
        "verifier",
        "final",
        "__end__",
    }


def test_build_entry_workflow_contains_intent_nodes() -> None:
    graph = workflow.build_entry_workflow()

    assert set(graph.get_graph().nodes) == {
        "__start__",
        "intent_router",
        "chat_responder",
        "__end__",
    }


@pytest.mark.parametrize(
    ("route", "expected_response", "expected_calls"),
    [
        ("chat", "Direct chat answer.", ["intent_router", "chat_responder"]),
        ("workflow", None, ["intent_router"]),
    ],
)
def test_entry_workflow_routes_chat_or_returns_for_main_workflow(
    route: str,
    expected_response: str | None,
    expected_calls: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_router(state: dict[str, Any]) -> dict[str, Any]:
        calls.append("intent_router")
        return {
            "intent_route": route,
            "intent_reason": "Test route",
            "intent_confidence": 0.9,
        }

    def fake_chat(state: dict[str, Any]) -> dict[str, Any]:
        calls.append("chat_responder")
        return {
            "chat_response": "Direct chat answer.",
            "final_answer": "Direct chat answer.",
        }

    monkeypatch.setattr(workflow, "intent_router_node", fake_router)
    monkeypatch.setattr(workflow, "chat_responder_node", fake_chat)

    result = workflow.build_entry_workflow().invoke({"task": "hello"})

    assert result["intent_route"] == route
    assert result.get("chat_response") == expected_response
    assert calls == expected_calls


def _fake_context_monitor(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_token_count": 100,
        "context_should_compress": False,
        "context_next_node": state.get("context_next_node", "verifier"),
    }


def test_workflow_reaches_final_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workflow,
        "planner_node",
        lambda state: {
            "plan_summary": "Create and verify the file.",
            "todos": [
                {
                    "id": "todo-1",
                    "content": "Create file",
                    "status": "pending",
                    "note": "",
                }
            ],
            "acceptance_criteria": ["File exists"],
            "verification_commands": ["test -f result.txt"],
        },
    )
    monkeypatch.setattr(
        workflow,
        "verifier_node",
        lambda state: {
            "passed": True,
            "attempts": state.get("attempts", 0) + 1,
            "verification_reason": "All checks passed.",
            "last_error": "",
            "context_next_node": "final",
        },
    )
    monkeypatch.setattr(workflow, "context_monitor_node", _fake_context_monitor)

    result = workflow.build_workflow().invoke(
        {
            "task": "Create result.txt",
            "attempts": 0,
            "max_attempts": 3,
        }
    )

    assert result["passed"] is True
    assert result["attempts"] == 1
    assert "Status: PASSED" in result["final_answer"]
    assert "All checks passed." in result["final_answer"]


def test_workflow_replans_until_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, int] = {
        "planner": 0,
        "verifier": 0,
    }

    def fake_planner(state: dict[str, Any]) -> dict[str, Any]:
        calls["planner"] += 1
        return {
            "plan_summary": f"Plan revision {calls['planner']}",
            "todos": [
                {
                    "id": "todo-1",
                    "content": "Fix implementation",
                    "status": "pending",
                    "note": "",
                }
            ],
            "acceptance_criteria": ["Tests pass"],
            "verification_commands": ["false"],
            "context_next_node": "verifier",
        }

    def fake_verifier(state: dict[str, Any]) -> dict[str, Any]:
        calls["verifier"] += 1
        attempts = state.get("attempts", 0) + 1
        return {
            "passed": False,
            "attempts": attempts,
            "verification_reason": "Tests still fail.",
            "last_error": "Fix the failing tests.",
            "context_next_node": (
                "final"
                if attempts >= state.get("max_attempts", 3)
                else "planner"
            ),
        }

    monkeypatch.setattr(workflow, "planner_node", fake_planner)
    monkeypatch.setattr(workflow, "verifier_node", fake_verifier)
    monkeypatch.setattr(workflow, "context_monitor_node", _fake_context_monitor)

    result = workflow.build_workflow().invoke(
        {
            "task": "Fix project",
            "attempts": 0,
            "max_attempts": 2,
        }
    )

    assert calls == {
        "planner": 2,
        "verifier": 2,
    }
    assert result["passed"] is False
    assert result["attempts"] == 2
    assert "Status: FAILED" in result["final_answer"]
    assert "Fix the failing tests." in result["final_answer"]


def test_complex_workflow_compresses_then_resumes_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_planner(state: dict[str, Any]) -> dict[str, Any]:
        calls.append("planner")
        return {"context_next_node": "verifier"}

    def fake_monitor(state: dict[str, Any]) -> dict[str, Any]:
        calls.append("context_monitor")
        return {
            "context_should_compress": len(calls) == 2,
            "context_next_node": state.get("context_next_node", "verifier"),
        }

    def fake_compressor(state: dict[str, Any]) -> dict[str, Any]:
        calls.append("context_compressor")
        return {"context_should_compress": False}

    def fake_verifier(state: dict[str, Any]) -> dict[str, Any]:
        calls.append("verifier")
        return {
            "passed": True,
            "attempts": 1,
            "verification_reason": "Passed after compression.",
            "context_next_node": "final",
        }

    monkeypatch.setattr(workflow, "planner_node", fake_planner)
    monkeypatch.setattr(workflow, "context_monitor_node", fake_monitor)
    monkeypatch.setattr(workflow, "context_compressor_node", fake_compressor)
    monkeypatch.setattr(workflow, "verifier_node", fake_verifier)

    result = workflow.build_complex_workflow().invoke(
        {
            "task": "Exercise compression route",
            "attempts": 0,
            "max_attempts": 3,
        }
    )

    assert calls == [
        "planner",
        "context_monitor",
        "context_compressor",
        "verifier",
        "context_monitor",
    ]
    assert result["passed"] is True
    assert "Status: PASSED" in result["final_answer"]
