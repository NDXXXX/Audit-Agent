import json
from pathlib import Path

import pytest

from audit_agent.core.state import RuntimeState
from audit_agent.core.trace import TraceRecorder, normalize_trace_mode


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (None, "summary"),
        ("", "summary"),
        ("invalid", "summary"),
        ("on", "summary"),
        (" SUMMARY ", "summary"),
        ("FULL", "full"),
        ("off", "off"),
    ],
)
def test_normalize_trace_mode(mode: str | None, expected: str) -> None:
    assert normalize_trace_mode(mode) == expected


def test_trace_records_events_statistics_and_human_timeline(
    tmp_path: Path,
) -> None:
    runtime = RuntimeState.create(
        tmp_path / "workspace",
        trace_mode="full",
        trace_id="trace-test-001",
    )
    recorder = TraceRecorder(runtime, task="Create an application")

    start_event = recorder.start(
        {
            "task": "Create an application",
            "runtime": runtime,
            "attempts": 0,
            "secret_context": "included only in full mode",
        }
    )
    recorder.record_custom_event(
        {
            "type": "custom",
            "node": "planner",
            "data": {
                "type": "tool_call",
                "agent": "codeAgent",
                "name": "bash",
                "args": {"command": "uv sync"},
            },
        }
    )
    recorder.record_custom_event(
        {
            "type": "tool_result",
            "agent": "codeAgent",
            "name": "bash",
            "result": {
                "ok": False,
                "requires_approval": True,
                "error": "Rejected",
            },
        }
    )
    recorder.record_custom_event(
        {
            "type": "handoff",
            "from": "planner",
            "to": "codeAgent",
            "instruction": "Implement the task",
        }
    )
    recorder.record_custom_event(
        {
            "type": "checkpoint_saved",
            "latest_node": "planner",
        }
    )
    recorder.record_graph_update(
        {
            "type": "node_update",
            "node": "planner",
            "data": {"plan_summary": "Build and test"},
        }
    )
    recorder.record_graph_update(
        {
            "type": "node_update",
            "node": "planner",
            "data": {"plan_summary": "Revise and test"},
        }
    )
    recorder.record_graph_update(
        {
            "type": "node_update",
            "node": "verifier",
            "data": {"passed": True},
        }
    )
    trace = recorder.end(
        status="passed",
        latest_node="final",
        final_state={"passed": True, "final_answer": "Done"},
    )

    assert start_event is not None
    assert start_event["type"] == "run_start"
    assert start_event["inputs"]["secret_context"] == (
        "included only in full mode"
    )
    assert trace is not None
    assert trace["trace_id"] == "trace-test-001"
    assert trace["task"] == "Create an application"
    assert trace["status"] == "passed"
    assert trace["latest_node"] == "final"
    assert trace["duration_ms"] >= 0
    assert trace["node_visits"] == {"planner": 2, "verifier": 1}
    assert trace["tool_calls"] == 1
    assert trace["failed_tool_calls"] == 1
    assert trace["approval_count"] == 1
    assert trace["checkpoint_count"] == 1
    assert trace["handoff_count"] == 1
    assert trace["timeline_omitted"] == 0

    assert recorder.root == (
        runtime.workspace / ".audit" / "traces" / "trace-test-001"
    )
    stored_trace = json.loads(
        (recorder.root / "trace.json").read_text(encoding="utf-8")
    )
    assert stored_trace == trace
    event_lines = (recorder.root / "events.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(event_lines) == 9
    assert json.loads(event_lines[0])["type"] == "run_start"
    assert json.loads(event_lines[-1])["type"] == "run_end"
    timeline = (recorder.root / "timeline.md").read_text(encoding="utf-8")
    assert "# Audit Agent Execution Trace" in timeline
    assert "Create an application" in timeline
    assert "tool_call" in timeline
    assert "planner" in timeline
    assert "status `passed`" in timeline


def test_summary_trace_limits_state_but_keeps_event_log(tmp_path: Path) -> None:
    runtime = RuntimeState.create(
        tmp_path / "workspace",
        trace_mode="summary",
        trace_id="summary-trace",
    )
    recorder = TraceRecorder(runtime)

    event = recorder.start(
        {
            "task": "Summarized task",
            "attempts": 1,
            "plan_summary": "Continue implementation",
            "large_internal_value": "x" * 10_000,
        }
    )
    trace = recorder.end(
        status="failed",
        latest_node="verifier",
        final_state={
            "passed": False,
            "last_error": "not retained in summary mode",
        },
    )

    assert event is not None
    assert event["inputs"] == {
        "task": "Summarized task",
        "attempts": 1,
        "plan_summary": "Continue implementation",
    }
    assert trace is not None
    assert (recorder.root / "events.jsonl").is_file()
    assert (recorder.root / "trace.json").is_file()
    assert (recorder.root / "timeline.md").is_file()


def test_trace_head_tail_window_reports_omitted_events(tmp_path: Path) -> None:
    recorder = TraceRecorder(
        RuntimeState.create(
            tmp_path / "workspace",
            trace_mode="full",
            trace_id="long-trace",
        ),
        task="Long trace",
    )
    recorder.start({"task": "Long trace"})
    for index in range(105):
        recorder.record_custom_event(
            {
                "type": "memory",
                "node": "planner",
                "index": index,
            }
        )

    trace = recorder.end(
        status="passed",
        latest_node="final",
        final_state={"passed": True},
    )

    assert trace is not None
    assert len(trace["timeline_head"]) == 20
    assert len(trace["timeline_tail"]) == 80
    assert trace["timeline_omitted"] == 7
    assert trace["timeline_head"][0]["type"] == "run_start"
    assert trace["timeline_tail"][-1]["type"] == "run_end"


def test_off_trace_mode_writes_nothing(tmp_path: Path) -> None:
    runtime = RuntimeState.create(
        tmp_path / "workspace",
        trace_mode="off",
        trace_id="disabled-trace",
    )
    recorder = TraceRecorder(runtime, task="Disabled")

    assert recorder.enabled is False
    assert recorder.start({"task": "Disabled"}) is None
    assert recorder.record_custom_event({"type": "tool_call"}) is None
    assert recorder.record_graph_update({"node": "planner"}) is None
    assert recorder.end(
        status="failed",
        latest_node=None,
        final_state={},
    ) is None
    assert not recorder.root.exists()


def test_trace_uses_resume_event_and_safe_trace_id(tmp_path: Path) -> None:
    runtime = RuntimeState.create(
        tmp_path / "workspace",
        trace_mode="full",
        trace_id="../unsafe trace/id",
    )
    recorder = TraceRecorder(runtime)

    event = recorder.start(
        {"task": "Resume task"},
        resumed=True,
        resume_event={"type": "checkpoint_resumed"},
    )

    assert recorder.trace_id == "unsafe-trace-id"
    assert recorder.root.is_relative_to(
        runtime.workspace / ".audit" / "traces"
    )
    assert event is not None
    assert event["resumed"] is True
    assert event["resume_event"] == {"type": "checkpoint_resumed"}


def test_runtime_state_normalizes_trace_configuration(tmp_path: Path) -> None:
    state = RuntimeState.create(
        tmp_path,
        trace_mode="FULL",
        trace_id="existing-trace",
    )

    assert state.trace_mode == "full"
    assert state.trace_id == "existing-trace"


def test_trace_end_is_idempotent_and_writes_one_run_end(tmp_path: Path) -> None:
    recorder = TraceRecorder(
        RuntimeState.create(
            tmp_path / "workspace",
            trace_mode="full",
            trace_id="idempotent-trace",
        ),
        task="Interrupt once",
    )
    recorder.start({"task": "Interrupt once"})

    first = recorder.end(
        status="interrupted",
        latest_node="planner",
        final_state={"passed": False},
    )
    second = recorder.end(
        status="failed",
        latest_node="verifier",
        final_state={"passed": False},
    )
    events = [
        json.loads(line)
        for line in (recorder.root / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert first == second
    assert first is not None and first["status"] == "interrupted"
    assert sum(event["type"] == "run_end" for event in events) == 1


def test_trace_counts_one_approval_even_with_reuse_and_tool_result(
    tmp_path: Path,
) -> None:
    recorder = TraceRecorder(
        RuntimeState.create(
            tmp_path / "workspace",
            trace_mode="full",
            trace_id="approval-trace",
        )
    )
    recorder.start({"task": "Approval count"})
    recorder.record_custom_event(
        {
            "type": "approval_requested",
            "approval_id": "approval-12345678",
        }
    )
    recorder.record_custom_event(
        {
            "type": "approval_decision",
            "approval_id": "approval-12345678",
            "approved": False,
            "reused": False,
        }
    )
    recorder.record_custom_event(
        {
            "type": "approval_decision",
            "approval_id": "approval-12345678",
            "approved": False,
            "reused": True,
        }
    )
    recorder.record_custom_event(
        {
            "type": "tool_result",
            "result": {
                "ok": False,
                "requires_approval": True,
                "approval_id": "approval-12345678",
            },
        }
    )

    trace = recorder.end(
        status="interrupted",
        latest_node="planner",
        final_state={},
    )

    assert trace is not None
    assert trace["approval_count"] == 1
