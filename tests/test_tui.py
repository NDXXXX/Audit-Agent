import threading
from pathlib import Path
from typing import Any

import pytest
from textual.binding import Binding
from textual.widgets import Input, Static

from audit_agent.cli.tui import app as tui_app_module
from audit_agent.cli.tui.approval import (
    ApprovalGate,
    ApprovalModal,
    ApprovalRequestedMessage,
)
from audit_agent.cli.tui.app import AgentEventMessage, AuditAgentTuiApp
from audit_agent.cli.tui.logo import (
    LOGO_ART,
    LOGO_EARS,
    LOGO_FACE,
    LOGO_STAGE,
    LOGO_TAG,
    LOGO_TITLE,
    AuditAgentLogo,
    render_logo,
)
from audit_agent.core.approval import ApprovalRequest


def _approval_request() -> ApprovalRequest:
    return ApprovalRequest(
        id="approval-test",
        command="python -m pip install example",
        risk_reason="Python package installation",
    )


def test_approval_gate_blocks_until_resolved(tmp_path: Path) -> None:
    gate = ApprovalGate(_approval_request(), tmp_path)
    decisions = []

    waiter = threading.Thread(target=lambda: decisions.append(gate.wait(1)))
    waiter.start()
    assert gate.resolved is False

    assert gate.resolve(True, "Approved in test") is True
    assert gate.resolve(False, "Too late") is False
    waiter.join(timeout=1)

    assert gate.resolved is True
    assert len(decisions) == 1
    assert decisions[0].approved is True
    assert decisions[0].reason == "Approved in test"


def test_approval_gate_wait_timeout(tmp_path: Path) -> None:
    gate = ApprovalGate(_approval_request(), tmp_path)

    with pytest.raises(TimeoutError, match="approval-test"):
        gate.wait(0.001)


def test_logo_art_and_boot_frames_render_the_cat_reveal() -> None:
    assert LOGO_ART == "\n".join(
        (LOGO_EARS, LOGO_FACE, LOGO_TAG, LOGO_TITLE, LOGO_STAGE)
    )
    assert "/\\_/\\" in LOGO_ART
    assert "( •ᴗ• )ฅ" in LOGO_ART
    assert "━━━ Audit Agent ━━━" in LOGO_ART
    assert "MultiAgent Coding Companion" in LOGO_ART
    assert "( -.- )" in render_logo(0).plain
    assert "( •ᴗ• )" in render_logo(4).plain
    assert "( •ᴗ• )ฅ" in render_logo(8).plain
    assert "╱" in render_logo(12).plain
    assert render_logo().plain == LOGO_ART


@pytest.mark.parametrize(
    ("state", "marker"),
    [
        ("idle", "( •ᴗ• )ฅ"),
        ("planner", "( •_• )"),
        ("tool", "⌨"),
        ("approval", "( !ᴗ! )"),
        ("success", "✓"),
        ("error", "( ×﹏× )"),
    ],
)
def test_logo_workflow_states_have_distinct_cat_frames(
    state: str,
    marker: str,
) -> None:
    rendered = render_logo(21, state=state)  # type: ignore[arg-type]
    assert marker in rendered.plain
    assert rendered.spans


def test_logo_rejects_unknown_state() -> None:
    with pytest.raises(ValueError, match="Unsupported Audit Agent logo state"):
        render_logo(state="unknown")  # type: ignore[arg-type]


def test_tui_has_terminal_safe_exit_shortcuts() -> None:
    bindings = {
        binding.key: binding
        for binding in AuditAgentTuiApp.BINDINGS
        if isinstance(binding, Binding)
    }

    assert {"ctrl+c", "ctrl+q", "f10"} <= bindings.keys()
    assert bindings["ctrl+c"].priority is True
    assert bindings["ctrl+q"].priority is True
    assert bindings["f10"].priority is True


@pytest.mark.asyncio
async def test_tui_mounts_required_layout_and_renders_agent_events(
    tmp_path: Path,
) -> None:
    app = AuditAgentTuiApp(
        session_workspace=tmp_path,
        checkpoint_mode="off",
        trace_mode="off",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.query_one("#task-input", Input)
        logo = app.query_one("#ascii-logo", AuditAgentLogo)
        assert app.query_one("#event-log")
        assert app.query_one("#status-bar", Static)

        app.post_message(
            AgentEventMessage(
                {
                    "type": "graph_event",
                    "event": {
                        "planner": {
                            "plan_summary": "Implement and test.",
                            "todos": [
                                {
                                    "id": "todo-1",
                                    "content": "Create app",
                                    "status": "completed",
                                    "note": "",
                                },
                                {
                                    "id": "todo-2",
                                    "content": "Run tests",
                                    "status": "in_progress",
                                    "note": "",
                                },
                            ],
                        }
                    },
                }
            )
        )
        app.post_message(
            AgentEventMessage(
                {
                    "type": "custom_event",
                    "event": {
                        "type": "tool_call",
                        "name": "FileWriteTool",
                        "args": {"file_path": "app.py"},
                    },
                }
            )
        )
        app.post_message(
            AgentEventMessage(
                {
                    "type": "graph_event",
                    "event": {
                        "final": {"final_answer": "Implementation complete."}
                    },
                }
            )
        )
        await pilot.pause()

        plan_text = str(app.query_one("#plan-panel", Static).render())
        assert "todo-1 ✅" in plan_text
        assert "todo-2 🔄" in plan_text
        assert app._last_final_answer == "Implementation complete."
        assert logo.logo_state == "success"


@pytest.mark.asyncio
async def test_tui_approval_message_opens_modal_and_y_approves(
    tmp_path: Path,
) -> None:
    app = AuditAgentTuiApp(
        session_workspace=tmp_path,
        checkpoint_mode="off",
        trace_mode="off",
    )
    gate = ApprovalGate(_approval_request(), tmp_path)

    async with app.run_test(size=(120, 40)) as pilot:
        app.post_message(ApprovalRequestedMessage(gate))
        await pilot.pause()

        assert isinstance(app.screen, ApprovalModal)
        assert app.query_one("#ascii-logo", AuditAgentLogo).logo_state == "approval"
        command = app.screen.query_one("#approval-command", Static)
        assert str(command.render()) == "python -m pip install example"

        await pilot.press("y")
        await pilot.pause()

        assert not isinstance(app.screen, ApprovalModal)
        decision = gate.wait(0.1)
        assert decision.approved is True
        assert "Approved by the user" in decision.reason
        assert app.query_one("#ascii-logo", AuditAgentLogo).logo_state == "tool"


@pytest.mark.asyncio
async def test_tui_input_runs_session_stream_in_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    received: dict[str, Any] = {}

    def fake_stream(task: str, **kwargs: Any) -> Any:
        received["task"] = task
        received.update(kwargs)
        yield {
            "type": "graph_event",
            "event": {
                "intent_router": {
                    "intent_route": "chat",
                    "intent_reason": "Greeting",
                    "intent_confidence": 0.9,
                }
            },
        }
        yield {
            "type": "graph_event",
            "event": {
                "chat_responder": {
                    "chat_response": "你好！",
                    "final_answer": "你好！",
                }
            },
        }

    monkeypatch.setattr(tui_app_module, "stream_session_events", fake_stream)
    app = AuditAgentTuiApp(
        session_workspace=tmp_path,
        max_attempts=5,
        checkpoint_mode="off",
        trace_mode="off",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.click("#task-input")
        await pilot.press(*"你好")
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert received["task"] == "你好"
        assert received["session_workspace"] == tmp_path.resolve()
        assert received["max_attempts"] == 5
        assert callable(received["approval_handler"])
        assert app._turn_running is False
        assert app.query_one("#task-input", Input).disabled is False
        assert app._last_final_answer == "你好！"
        assert app.query_one("#ascii-logo", AuditAgentLogo).logo_state == "success"


@pytest.mark.asyncio
async def test_tui_maps_agent_events_to_logo_states(tmp_path: Path) -> None:
    app = AuditAgentTuiApp(
        session_workspace=tmp_path,
        checkpoint_mode="off",
        trace_mode="off",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        logo = app.query_one("#ascii-logo", AuditAgentLogo)

        app._render_graph_update("planner", {"todos": []})
        assert logo.logo_state == "planner"

        app._render_custom_payload(
            {"type": "tool_call", "name": "BashTool", "args": {}}
        )
        assert logo.logo_state == "tool"

        app._render_custom_payload(
            {"type": "approval_requested", "risk_reason": "download"}
        )
        assert logo.logo_state == "approval"

        app._render_graph_update("verifier", {"passed": False})
        assert logo.logo_state == "error"

        app._render_graph_update("verifier", {"passed": True})
        assert logo.logo_state == "success"
        await pilot.pause()
