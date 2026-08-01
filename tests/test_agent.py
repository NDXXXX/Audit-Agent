import importlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ddclaw.core import agent as agent_module
from ddclaw.core.approval import ApprovalRequest, RunInterrupted
from ddclaw.core.checkpoint import CheckpointManager
from ddclaw.core.session import SESSION_FILE, SESSION_ROOT
from ddclaw.core.state import RuntimeState


class FakeWorkflow:
    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.inputs: dict[str, Any] | None = None
        self.stream_mode: list[str] | None = None

    def stream(
        self,
        inputs: dict[str, Any],
        *,
        stream_mode: list[str],
    ) -> Any:
        self.inputs = inputs
        self.stream_mode = stream_mode
        yield from self.events


class InterruptWorkflow:
    def stream(
        self,
        inputs: dict[str, Any],
        *,
        stream_mode: list[str],
    ) -> Any:
        yield "updates", {"planner": {"plan_summary": "Interrupted plan"}}
        raise KeyboardInterrupt


class ApprovalInterruptWorkflow:
    def stream(
        self,
        inputs: dict[str, Any],
        *,
        stream_mode: list[str],
    ) -> Any:
        raise RunInterrupted("Approval prompt interrupted")
        yield  # pragma: no cover


def test_stream_agent_events_normalizes_tuple_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workflow = FakeWorkflow(
        [
            (
                "updates",
                {
                    "planner": {
                        "plan_summary": "Create a greeting.",
                        "todos": [],
                    }
                },
            ),
            (
                "custom",
                {
                    "type": "tool_call",
                    "agent": "codeAgent",
                    "name": "file_write",
                    "args": {"file_path": "hello.txt"},
                },
            ),
            (
                "updates",
                {"verifier": {"passed": True, "attempts": 1}},
            ),
            (
                "updates",
                {"final": {"final_answer": "Task passed."}},
            ),
        ]
    )
    monkeypatch.setattr(agent_module, "build_workflow", lambda: workflow)

    events = list(
        agent_module.stream_agent_events(
            "Create a greeting file",
            workspace=tmp_path / "workspace",
            max_attempts=4,
        )
    )

    assert events == [
        {
            "type": "graph_event",
            "event": {
                "planner": {
                    "plan_summary": "Create a greeting.",
                    "todos": [],
                },
            },
        },
        {
            "type": "custom_event",
            "event": {
                "type": "tool_call",
                "agent": "codeAgent",
                "name": "file_write",
                "args": {"file_path": "hello.txt"},
            },
        },
        {
            "type": "graph_event",
            "event": {
                "verifier": {"passed": True, "attempts": 1},
            },
        },
        {
            "type": "graph_event",
            "event": {
                "final": {"final_answer": "Task passed."},
            },
        },
    ]
    assert workflow.stream_mode == ["updates", "custom"]
    assert workflow.inputs is not None
    assert workflow.inputs["task"] == "Create a greeting file"
    assert workflow.inputs["max_attempts"] == 4
    assert workflow.inputs["attempts"] == 0
    assert workflow.inputs["research_notes"] == ""
    assert workflow.inputs["sources"] == []
    assert workflow.inputs["agent_handoffs"] == []
    assert isinstance(workflow.inputs["runtime"], RuntimeState)
    assert workflow.inputs["runtime"].workspace == (tmp_path / "workspace").resolve()
    assert workflow.inputs["runtime"].approval_mode == "inline"
    assert workflow.inputs["runtime"].checkpoint_mode == "light"
    assert workflow.inputs["runtime"].trace_mode == "summary"
    checkpoint = json.loads(
        (
            workflow.inputs["runtime"].workspace
            / ".ddclaw"
            / "checkpoints"
            / "checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    assert checkpoint["status"] == "finished"
    assert checkpoint["latest_node"] == "final"
    trace_files = list(
        (
            workflow.inputs["runtime"].workspace / ".ddclaw" / "traces"
        ).glob("*/trace.json")
    )
    assert len(trace_files) == 1
    trace = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert trace["status"] == "finished"
    assert trace["node_visits"] == {
        "planner": 1,
        "verifier": 1,
        "final": 1,
    }


def test_stream_agent_events_normalizes_v2_dictionary_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workflow = FakeWorkflow(
        [
            {
                "type": "updates",
                "data": {"planner": {"plan_summary": "Plan"}},
                "ns": [],
            },
            {
                "type": "custom",
                "data": {"type": "tool_result", "result": "ok"},
                "ns": ["planner"],
            },
        ]
    )
    monkeypatch.setattr(agent_module, "build_workflow", lambda: workflow)

    events = list(
        agent_module.stream_agent_events(
            "task",
            workspace=tmp_path,
        )
    )

    assert events == [
        {
            "type": "graph_event",
            "event": {"planner": {"plan_summary": "Plan"}},
        },
        {
            "type": "custom_event",
            "event": {"type": "tool_result", "result": "ok"},
        },
    ]


def test_stream_agent_events_validates_max_attempts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        list(
            agent_module.stream_agent_events(
                "task",
                workspace=tmp_path,
                max_attempts=0,
            )
        )


def test_stream_agent_events_passes_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workflow = FakeWorkflow([])
    monkeypatch.setattr(agent_module, "build_workflow", lambda: workflow)
    handler = lambda request: None

    events = list(
        agent_module.stream_agent_events(
            "configured task",
            workspace=tmp_path / "workspace",
            approval_mode="auto",
            approval_handler=handler,
            checkpoint_mode="off",
            trace_mode="off",
        )
    )

    assert events == []
    assert workflow.inputs is not None
    runtime = workflow.inputs["runtime"]
    assert runtime.approval_mode == "auto"
    assert runtime.approval_handler is handler
    assert runtime.checkpoint_mode == "off"
    assert runtime.trace_mode == "off"
    assert not (runtime.workspace / ".ddclaw").exists()


def test_stream_agent_events_checkpoints_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        agent_module,
        "build_workflow",
        lambda: InterruptWorkflow(),
    )
    workspace = tmp_path / "workspace"

    with pytest.raises(KeyboardInterrupt):
        list(
            agent_module.stream_agent_events(
                "interrupt task",
                workspace=workspace,
                checkpoint_mode="light",
                trace_mode="full",
            )
        )

    checkpoint = json.loads(
        (
            workspace / ".ddclaw" / "checkpoints" / "checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    assert checkpoint["status"] == "interrupted"
    assert checkpoint["latest_node"] == "planner"
    assert checkpoint["state_summary"]["plan_summary"] == "Interrupted plan"
    trace_path = next((workspace / ".ddclaw" / "traces").glob("*/trace.json"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["status"] == "interrupted"
    assert trace["latest_node"] == "planner"


def test_stream_agent_events_resumes_checkpoint_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resume_workspace = tmp_path / "resume-workspace"
    original_runtime = RuntimeState.create(
        resume_workspace,
        checkpoint_mode="light",
        trace_mode="off",
    )
    (resume_workspace / "saved.txt").write_text("saved", encoding="utf-8")
    CheckpointManager(original_runtime, task="saved task").save(
        {
            "task": "saved task",
            "runtime": original_runtime,
            "messages": [],
            "todos": [],
            "attempts": 2,
            "passed": False,
            "context_summary": "Continue from the saved plan.",
            "context_next_node": "planner",
        },
        status="interrupted",
        latest_node="verifier",
    )
    workflow = FakeWorkflow(
        [("updates", {"final": {"final_answer": "Resumed."}})]
    )
    monkeypatch.setattr(agent_module, "build_workflow", lambda: workflow)

    events = list(
        agent_module.stream_agent_events(
            "resumed task",
            workspace=tmp_path / "ignored-workspace",
            resume_workspace=resume_workspace,
            checkpoint_mode="light",
            trace_mode="off",
            max_attempts=5,
        )
    )

    assert events == [
        {
            "type": "graph_event",
            "event": {"final": {"final_answer": "Resumed."}},
        }
    ]
    assert workflow.inputs is not None
    assert workflow.inputs["runtime"].workspace == resume_workspace.resolve()
    assert workflow.inputs["task"] == "resumed task"
    assert workflow.inputs["attempts"] == 2
    assert workflow.inputs["max_attempts"] == 5
    assert "Continue from the saved plan" in workflow.inputs["messages"][0].content


def test_stream_session_events_routes_chat_and_persists_both_turns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "README.md").parent.mkdir(parents=True)
    (workspace / "README.md").write_text("project", encoding="utf-8")
    entry = FakeWorkflow(
        [
            (
                "updates",
                {
                    "intent_router": {
                        "intent_route": "chat",
                        "intent_reason": "Greeting",
                        "intent_confidence": 0.95,
                    }
                },
            ),
            (
                "updates",
                {
                    "chat_responder": {
                        "chat_response": "你好，我是 DDclaw。",
                        "final_answer": "你好，我是 DDclaw。",
                    }
                },
            ),
        ]
    )
    monkeypatch.setattr(agent_module, "build_entry_workflow", lambda: entry)

    def unexpected_complex_workflow() -> Any:
        raise AssertionError("chat route must not run the complex workflow")

    monkeypatch.setattr(
        agent_module,
        "build_complex_workflow",
        unexpected_complex_workflow,
    )

    events = list(
        agent_module.stream_session_events(
            "你好",
            session_workspace=workspace,
            checkpoint_mode="off",
            trace_mode="off",
        )
    )

    assert events == [
        {
            "type": "graph_event",
            "event": {
                "intent_router": {
                    "intent_route": "chat",
                    "intent_reason": "Greeting",
                    "intent_confidence": 0.95,
                }
            },
        },
        {
            "type": "graph_event",
            "event": {
                "chat_responder": {
                    "chat_response": "你好，我是 DDclaw。",
                    "final_answer": "你好，我是 DDclaw。",
                }
            },
        },
    ]
    assert entry.inputs is not None
    assert entry.inputs["session_turn"] == 1
    assert "README.md" in entry.inputs["session_context"]
    assert "Turn 1 [user]: 你好" in entry.inputs["session_context"]
    stored = json.loads(
        (workspace / SESSION_ROOT / SESSION_FILE).read_text(encoding="utf-8")
    )
    assert stored["turn_index"] == 1
    assert [item["role"] for item in stored["recent_turns"]] == [
        "user",
        "assistant",
    ]
    assert stored["recent_turns"][1]["route"] == "chat"
    assert stored["recent_turns"][1]["content"] == "你好，我是 DDclaw。"


def test_stream_session_events_hands_workflow_state_to_complex_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entry = FakeWorkflow(
        [
            (
                "updates",
                {
                    "intent_router": {
                        "intent_route": "workflow",
                        "intent_reason": "Needs files",
                        "intent_confidence": 0.99,
                    }
                },
            )
        ]
    )
    complex_workflow = FakeWorkflow(
        [
            (
                "updates",
                {
                    "planner": {
                        "plan_summary": "Create the requested file.",
                        "code_agent_summary": "Created hello.py.",
                    }
                },
            ),
            (
                "updates",
                {
                    "final": {
                        "passed": True,
                        "final_answer": "Task completed.",
                    }
                },
            ),
        ]
    )
    monkeypatch.setattr(agent_module, "build_entry_workflow", lambda: entry)
    monkeypatch.setattr(
        agent_module,
        "build_complex_workflow",
        lambda: complex_workflow,
    )
    workspace = tmp_path / "workspace"

    events = list(
        agent_module.stream_session_events(
            "创建 hello.py",
            session_workspace=workspace,
            max_attempts=4,
            checkpoint_mode="off",
            trace_mode="off",
        )
    )

    assert [next(iter(event["event"])) for event in events] == [
        "intent_router",
        "planner",
        "final",
    ]
    assert complex_workflow.inputs is not None
    assert complex_workflow.inputs["intent_route"] == "workflow"
    assert complex_workflow.inputs["session_turn"] == 1
    assert complex_workflow.inputs["max_attempts"] == 4
    stored = json.loads(
        (workspace / SESSION_ROOT / SESSION_FILE).read_text(encoding="utf-8")
    )
    assert stored["recent_turns"][-1]["route"] == "workflow"
    assert stored["recent_turns"][-1]["content"] == "Task completed."
    assert stored["recent_turns"][-1]["summary"] == "Created hello.py."


def test_stream_session_events_carries_prior_turns_into_next_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_inputs: list[dict[str, Any]] = []

    class CapturingChatWorkflow:
        def stream(
            self,
            inputs: dict[str, Any],
            *,
            stream_mode: list[str],
        ) -> Any:
            captured_inputs.append(inputs)
            yield (
                "updates",
                {
                    "intent_router": {
                        "intent_route": "chat",
                        "intent_reason": "Conversation",
                        "intent_confidence": 0.9,
                    }
                },
            )
            yield (
                "updates",
                {
                    "chat_responder": {
                        "chat_response": f"reply-{len(captured_inputs)}",
                        "final_answer": f"reply-{len(captured_inputs)}",
                    }
                },
            )

    monkeypatch.setattr(
        agent_module,
        "build_entry_workflow",
        CapturingChatWorkflow,
    )
    workspace = tmp_path / "workspace"

    list(
        agent_module.stream_session_events(
            "第一条消息",
            session_workspace=workspace,
            checkpoint_mode="off",
            trace_mode="off",
        )
    )
    list(
        agent_module.stream_session_events(
            "继续聊",
            session_workspace=workspace,
            checkpoint_mode="off",
            trace_mode="off",
        )
    )

    second_context = captured_inputs[1]["session_context"]
    assert captured_inputs[1]["session_turn"] == 2
    assert "第一条消息" in second_context
    assert "reply-1" in second_context
    assert "继续聊" in second_context


def test_stream_session_events_validates_max_attempts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        list(
            agent_module.stream_session_events(
                "hello",
                session_workspace=tmp_path,
                max_attempts=0,
            )
        )


def test_cli_renders_stage3_nodes_handoffs_and_specialist_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_module = importlib.import_module("ddclaw.cli.app")
    received: dict[str, Any] = {}

    def fake_events(*args: Any, **kwargs: Any) -> Any:
        received.update(kwargs)
        yield {
            "type": "node_update",
            "node": "planner",
            "data": {
                "plan_summary": "Create a greeting.",
                "todos": [
                    {
                        "id": "1",
                        "content": "Write hello.txt",
                        "status": "pending",
                        "note": "",
                    }
                ],
                "acceptance_criteria": ["hello.txt exists"],
                "verification_commands": ["test -f hello.txt"],
                "code_agent_summary": "Created hello.txt.",
                "agent_handoffs": [
                    {
                        "from_agent": "planner",
                        "to_agent": "codeAgent",
                        "instruction": "Create the greeting file.",
                        "result": "Created hello.txt.",
                    }
                ],
            },
        }
        yield {
            "type": "custom",
            "node": "planner",
            "data": {
                "type": "handoff",
                "from": "planner",
                "to": "codeAgent",
                "instruction": "Create the greeting file.",
            },
        }
        yield {
            "type": "custom",
            "node": "planner",
            "data": {
                "type": "tool_call",
                "agent": "codeAgent",
                "name": "file_write",
                "args": {"file_path": "hello.txt"},
            },
        }
        yield {
            "type": "custom",
            "node": "planner",
            "data": {
                "type": "tool_result",
                "agent": "codeAgent",
                "name": "file_write",
                "result": "Wrote hello.txt",
            },
        }
        yield {
            "type": "node_update",
            "node": "verifier",
            "data": {
                "passed": True,
                "attempts": 1,
                "verification_reason": "All checks passed.",
                "verification_checks": [
                    {
                        "name": "file exists",
                        "passed": True,
                        "detail": "Found hello.txt",
                    }
                ],
                "verification_results": [],
            },
        }
        yield {
            "type": "node_update",
            "node": "final",
            "data": {"final_answer": "Created `hello.txt`."},
        }

    monkeypatch.setattr(app_module, "stream_agent_events", fake_events)

    result = CliRunner().invoke(
        app_module.app,
        [
            "Create a greeting",
            "--workspace",
            str(tmp_path),
            "--max-attempts",
            "5",
        ],
    )

    assert result.exit_code == 0
    assert "📋 Planner" in result.stdout
    assert "planner → codeAgent" in result.stdout
    assert "🔧 codeAgent" in result.stdout
    assert "✅ Verifier" in result.stdout
    assert "📝 Final" in result.stdout
    assert "file_write" in result.stdout
    assert "Wrote hello.txt" in result.stdout
    assert "Created hello.txt." in result.stdout
    assert received["workspace"] == tmp_path
    assert received["max_attempts"] == 5
    assert received["approval_mode"] == "inline"
    assert callable(received["approval_handler"])
    assert received["checkpoint_mode"] == "light"
    assert received["trace_mode"] == "on"
    assert received["resume_workspace"] is None


def test_cli_passes_runtime_modes_and_resume_without_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_module = importlib.import_module("ddclaw.cli.app")
    received: dict[str, Any] = {}

    def fake_events(task: str, **kwargs: Any) -> Any:
        received["task"] = task
        received.update(kwargs)
        yield {
            "type": "graph_event",
            "event": {"final": {"final_answer": "Resumed task."}},
        }

    monkeypatch.setattr(app_module, "stream_agent_events", fake_events)
    result = CliRunner().invoke(
        app_module.app,
        [
            "--resume",
            str(tmp_path),
            "--approval-mode",
            "deny",
            "--checkpoint-mode",
            "strict",
            "--trace-mode",
            "off",
            "--max-attempts",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert "📝 Final" in result.stdout
    assert received == {
        "task": "",
        "workspace": tmp_path,
        "max_attempts": 7,
        "approval_mode": "deny",
        "approval_handler": None,
        "checkpoint_mode": "strict",
        "resume_workspace": tmp_path,
        "trace_mode": "off",
    }


def test_cli_requires_task_without_resume(tmp_path: Path) -> None:
    app_module = importlib.import_module("ddclaw.cli.app")

    result = CliRunner().invoke(
        app_module.app,
        ["--workspace", str(tmp_path)],
    )

    assert result.exit_code == 2
    assert "Provide a task or use --resume" in result.output


def test_cli_inline_approval_handler_returns_user_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_module = importlib.import_module("ddclaw.cli.app")
    monkeypatch.setattr(app_module.typer, "confirm", lambda *args, **kwargs: True)

    decision = app_module._inline_approval_handler(
        ApprovalRequest(
            id="approval-12345678",
            command="uv sync",
            risk_reason="Dependency synchronization with uv sync",
        )
    )

    assert decision.approved is True
    assert decision.reason == "Approved interactively by the user."


def test_closing_event_generator_finalizes_interrupted_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workflow = FakeWorkflow(
        [
            ("updates", {"planner": {"plan_summary": "In progress"}}),
            ("updates", {"final": {"final_answer": "Should not finish"}}),
        ]
    )
    monkeypatch.setattr(agent_module, "build_workflow", lambda: workflow)
    workspace = tmp_path / "workspace"
    events = agent_module.stream_agent_events(
        "close the generator",
        workspace=workspace,
        checkpoint_mode="strict",
        trace_mode="full",
    )

    first = next(events)
    events.close()

    assert first["type"] == "graph_event"
    checkpoint_root = workspace / ".ddclaw" / "checkpoints"
    checkpoint = json.loads(
        (checkpoint_root / "checkpoint.json").read_text(encoding="utf-8")
    )
    lease = json.loads(
        (checkpoint_root / "run.json").read_text(encoding="utf-8")
    )
    assert checkpoint["status"] == "interrupted"
    assert checkpoint["latest_node"] == "planner"
    assert lease["status"] == "interrupted"
    trace_events_path = next(
        (workspace / ".ddclaw" / "traces").glob("*/events.jsonl")
    )
    trace_events = [
        json.loads(line)
        for line in trace_events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(event["type"] == "run_end" for event in trace_events) == 1
    assert trace_events[-1]["status"] == "interrupted"


def test_approval_interrupt_finalizes_checkpoint_and_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        agent_module,
        "build_workflow",
        lambda: ApprovalInterruptWorkflow(),
    )
    workspace = tmp_path / "workspace"

    with pytest.raises(RunInterrupted):
        list(
            agent_module.stream_agent_events(
                "interrupt approval",
                workspace=workspace,
                checkpoint_mode="strict",
                trace_mode="full",
            )
        )

    checkpoint = json.loads(
        (workspace / ".ddclaw" / "checkpoints" / "checkpoint.json")
        .read_text(encoding="utf-8")
    )
    trace_path = next((workspace / ".ddclaw" / "traces").glob("*/trace.json"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert checkpoint["status"] == "interrupted"
    assert trace["status"] == "interrupted"


def test_strict_custom_event_uses_quick_checkpoint_without_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workflow = FakeWorkflow(
        [
            (
                "custom",
                {"type": "tool_result", "name": "bash", "result": {"ok": True}},
            )
        ]
    )
    monkeypatch.setattr(agent_module, "build_workflow", lambda: workflow)
    snapshots: list[bool] = []
    original_save = CheckpointManager.save

    def recording_save(self: CheckpointManager, *args: Any, **kwargs: Any) -> Any:
        snapshots.append(bool(kwargs.get("snapshot", True)))
        return original_save(self, *args, **kwargs)

    monkeypatch.setattr(CheckpointManager, "save", recording_save)

    list(
        agent_module.stream_agent_events(
            "quick strict checkpoints",
            workspace=tmp_path / "workspace",
            checkpoint_mode="strict",
            trace_mode="off",
        )
    )

    assert snapshots == [True, False, True]


def test_cli_converts_prompt_abort_into_clean_interruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_module = importlib.import_module("ddclaw.cli.app")

    def abort(*args: Any, **kwargs: Any) -> bool:
        raise typer.Abort()

    import typer

    monkeypatch.setattr(app_module.typer, "confirm", abort)

    with pytest.raises(RunInterrupted):
        app_module._inline_approval_handler(
            ApprovalRequest(
                id="approval-aborted",
                command="uv sync",
                risk_reason="Dependency synchronization with uv sync",
            )
        )


def test_interrupt_during_initial_checkpoint_is_finalized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = agent_module._save_and_trace_checkpoint
    calls = 0

    def interrupt_first_save(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        original(*args, **kwargs)

    monkeypatch.setattr(
        agent_module,
        "_save_and_trace_checkpoint",
        interrupt_first_save,
    )
    workspace = tmp_path / "workspace"

    with pytest.raises(KeyboardInterrupt):
        list(
            agent_module.stream_agent_events(
                "interrupt initial snapshot",
                workspace=workspace,
                checkpoint_mode="strict",
                trace_mode="full",
            )
        )

    checkpoint = json.loads(
        (workspace / ".ddclaw" / "checkpoints" / "checkpoint.json")
        .read_text(encoding="utf-8")
    )
    assert checkpoint["status"] == "interrupted"
    assert checkpoint["latest_node"] == "start"


def test_cli_reports_clean_exit_code_for_run_interruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_module = importlib.import_module("ddclaw.cli.app")

    def interrupted_stream(*args: Any, **kwargs: Any) -> Any:
        raise RunInterrupted("Stopped at approval")
        yield  # pragma: no cover

    monkeypatch.setattr(app_module, "stream_agent_events", interrupted_stream)
    result = CliRunner().invoke(
        app_module.app,
        ["task", "--workspace", str(tmp_path)],
    )

    assert result.exit_code == 130
    assert "Run interrupted" in result.output
    assert "ddclaw --resume" in result.output
