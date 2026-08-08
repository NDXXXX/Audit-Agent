"""Interactive Textual application for multi-turn Audit Agent sessions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from threading import Lock
from typing import Any

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Footer, Header, Input, RichLog, Static

from audit_agent.cli.tui.approval import (
    ApprovalGate,
    ApprovalModal,
    ApprovalRequestedMessage,
)
from audit_agent.cli.tui.logo import AuditAgentLogo, LogoState
from audit_agent.core.agent import stream_session_events
from audit_agent.core.approval import ApprovalDecision, ApprovalRequest
from audit_agent.core.session import load_or_create_session


class AgentEventMessage(Message):
    """One event emitted by ``stream_session_events`` in the Agent thread."""

    def __init__(self, event: Mapping[str, Any]) -> None:
        self.event = dict(event)
        super().__init__()


class AgentRunFinishedMessage(Message):
    """Signal that one submitted session turn stopped running."""

    def __init__(self, error: str = "") -> None:
        self.error = error
        super().__init__()


class AuditAgentTuiApp(App[None]):
    """Persistent multi-turn Audit Agent terminal interface."""

    TITLE = "🐾 Audit Agent"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("f10", "quit", "Quit", show=False, priority=True),
        Binding("ctrl+l", "clear_events", "Clear events"),
    ]

    CSS = """
    Screen {
        layout: vertical;
        background: $background;
    }

    Header {
        dock: top;
    }

    #status-bar {
        height: 1;
        padding: 0 2;
        color: $text-muted;
        background: $panel;
    }

    #plan-panel {
        min-height: 3;
        max-height: 8;
        padding: 1 2;
        margin: 1 1 0 1;
        border: round $primary;
        background: $surface;
    }

    #event-log {
        height: 1fr;
        margin: 1;
        padding: 0 1;
        border: round $secondary;
        background: $surface;
    }

    #task-input {
        dock: bottom;
        margin: 0 1 1 1;
        border: tall $accent;
    }

    Footer {
        dock: bottom;
    }
    """

    def __init__(
        self,
        *,
        session_workspace: str | PathLike[str] | None = None,
        max_attempts: int = 3,
        approval_mode: str = "inline",
        checkpoint_mode: str = "light",
        trace_mode: str = "on",
    ) -> None:
        super().__init__()
        if max_attempts < 1:
            raise ValueError("max_attempts must be greater than or equal to 1")
        self.session_workspace = Path(
            session_workspace or Path.cwd() / ".audit-workspace"
        ).expanduser().resolve()
        self.max_attempts = max_attempts
        self.approval_mode = approval_mode
        self.checkpoint_mode = checkpoint_mode
        self.trace_mode = trace_mode
        self._session = load_or_create_session(self.session_workspace)
        self.sub_title = f"session: {self._short_session_id}"
        self._turn_running = False
        self._pending_gates: dict[str, ApprovalGate] = {}
        self._gate_lock = Lock()
        self._last_final_answer = ""
        self._logo_reset_timer: Timer | None = None

    @property
    def _short_session_id(self) -> str:
        return str(self._session.get("session_id", "unknown"))[:8]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield AuditAgentLogo(id="ascii-logo")
        yield Static(
            self._status_text("Ready"),
            id="status-bar",
            markup=False,
        )
        yield Static("[Plan] No active plan", id="plan-panel", markup=False)
        yield RichLog(
            id="event-log",
            wrap=True,
            markup=False,
            auto_scroll=True,
            max_lines=2_000,
        )
        yield Input(
            placeholder="💬 Input — describe a task or ask a question",
            id="task-input",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#task-input", Input).focus()
        self._write_event(
            "🐾",
            f"Session {self._short_session_id} ready in {self.session_workspace}",
            style="bold cyan",
        )

    @on(Input.Submitted, "#task-input")
    def submit_task(self, event: Input.Submitted) -> None:
        task = event.value.strip()
        if not task:
            return
        if self._turn_running:
            self.notify("A session turn is already running.", severity="warning")
            return

        event.input.value = ""
        event.input.disabled = True
        self._cancel_logo_reset()
        self._set_logo_state("planner")
        self._turn_running = True
        self._last_final_answer = ""
        self._set_status("Running")
        self._write_event("💬", task, style="bold")
        self.run_worker(
            lambda: self._run_session_turn(task),
            name=f"session-turn-{self._session.get('turn_index', 0) + 1}",
            group="session-turn",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _run_session_turn(self, task: str) -> None:
        error = ""
        try:
            for event in stream_session_events(
                task,
                session_workspace=self.session_workspace,
                max_attempts=self.max_attempts,
                approval_mode=self.approval_mode,
                approval_handler=(
                    self._approval_handler
                    if self.approval_mode == "inline"
                    else None
                ),
                checkpoint_mode=self.checkpoint_mode,
                trace_mode=self.trace_mode,
            ):
                self.post_message(AgentEventMessage(event))
        except Exception as exc:  # surfaced in the event log on the UI thread
            error = f"{type(exc).__name__}: {exc}"
        finally:
            self.post_message(AgentRunFinishedMessage(error))

    def _approval_handler(
        self,
        request: ApprovalRequest,
    ) -> ApprovalDecision:
        gate = ApprovalGate(request, self.session_workspace)
        with self._gate_lock:
            self._pending_gates[request.id] = gate
        try:
            if not self.post_message(ApprovalRequestedMessage(gate)):
                gate.resolve(False, "The Audit Agent TUI is no longer running.")
            return gate.wait()
        finally:
            with self._gate_lock:
                self._pending_gates.pop(request.id, None)

    def on_agent_event_message(self, message: AgentEventMessage) -> None:
        self._render_agent_event(message.event)

    def on_agent_run_finished_message(
        self,
        message: AgentRunFinishedMessage,
    ) -> None:
        self._turn_running = False
        task_input = self.query_one("#task-input", Input)
        task_input.disabled = False
        task_input.focus()
        self._session = load_or_create_session(self.session_workspace)
        self.sub_title = f"session: {self._short_session_id}"
        if message.error:
            self._set_logo_state("error")
            self._set_status("Failed")
            self._write_event("❌", message.error, style="bold red")
        else:
            self._set_status("Ready")
            logo = self.query_one("#ascii-logo", AuditAgentLogo)
            if logo.logo_state not in {"success", "error"}:
                logo.set_state("success")
        self._schedule_logo_idle()

    def on_approval_requested_message(
        self,
        message: ApprovalRequestedMessage,
    ) -> None:
        gate = message.gate
        self._set_logo_state("approval")
        self._set_status("Waiting for approval")
        self._write_event(
            "⚠️",
            f"Approval requested: {gate.request.risk_reason}",
            style="bold yellow",
        )

        def resolve(approved: bool | None) -> None:
            accepted = bool(approved)
            gate.resolve(
                accepted,
                (
                    "Approved by the user in the Audit Agent TUI."
                    if accepted
                    else "Denied by the user in the Audit Agent TUI."
                ),
            )
            self._set_logo_state("tool" if accepted else "error")
            self._set_status("Running")
            self._write_event(
                "✅" if accepted else "⛔",
                "Command approved." if accepted else "Command denied.",
                style="green" if accepted else "red",
            )

        self.push_screen(
            ApprovalModal(gate.request, gate.workspace),
            resolve,
        )

    def _render_agent_event(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type == "custom_event":
            payload = event.get("event")
            if isinstance(payload, Mapping):
                self._render_custom_payload(payload)
            return
        if event_type == "graph_event":
            graph_event = event.get("event")
            if isinstance(graph_event, Mapping):
                for node, update in graph_event.items():
                    self._render_graph_update(
                        str(node),
                        update if isinstance(update, Mapping) else {},
                    )
            return
        self._render_custom_payload(event)

    def _render_graph_update(
        self,
        node: str,
        update: Mapping[str, Any],
    ) -> None:
        if node == "intent_router":
            route = update.get("intent_route", "workflow")
            reason = update.get("intent_reason", "")
            self._set_logo_state("idle" if route == "chat" else "planner")
            self._write_event("🧭", f"Route: {route} — {reason}")
        elif node == "planner":
            self._set_logo_state("planner")
            self._update_plan(update.get("todos", []))
            summary = update.get("plan_summary")
            if summary:
                self._write_event("📋", str(summary), style="bold blue")
        elif node == "verifier":
            passed = bool(update.get("passed"))
            self._set_logo_state("success" if passed else "error")
            detail = update.get("verification_reason") or update.get("last_error")
            self._write_event(
                "✅" if passed else "❌",
                f"Verifier: {detail or ('passed' if passed else 'failed')}",
                style="green" if passed else "red",
            )
        elif node in {"chat_responder", "final"}:
            content = update.get("final_answer") or update.get("chat_response")
            if content:
                self._set_logo_state("success")
                self._show_final_answer(str(content))
        elif node == "context_compressor":
            self._set_logo_state("planner")
            self._write_event("🗜️", "Context compressed for the next stage.")

    def _render_custom_payload(self, payload: Mapping[str, Any]) -> None:
        payload_type = str(payload.get("type") or "")
        if payload_type == "plan_snapshot":
            self._set_logo_state("planner")
            self._update_plan(payload.get("todos", []))
        elif payload_type == "tool_call":
            self._set_logo_state("tool")
            name = str(payload.get("name") or "Tool")
            args = payload.get("args")
            self._write_event(
                "🔧",
                f"{name} → {_event_target(args)}",
                style="cyan",
            )
        elif payload_type == "tool_result":
            name = str(payload.get("name") or "Tool")
            result = payload.get("result")
            ok = _result_ok(payload, result)
            self._set_logo_state("tool" if ok else "error")
            self._write_event(
                "✅" if ok else "❌",
                f"{name}: {_compact_value(result)}",
                style="green" if ok else "red",
            )
        elif payload_type == "handoff":
            self._set_logo_state("planner")
            self._write_event(
                "🔄",
                f"Handoff: {payload.get('from', 'planner')} → "
                f"{payload.get('to', 'agent')} — "
                f"{payload.get('instruction', '')}",
                style="magenta",
            )
        elif payload_type == "search_results":
            self._set_logo_state("tool")
            query = payload.get("query") or payload.get("instruction") or "search"
            self._write_event("🔍", f"WebSearchTool: {query}", style="magenta")
        elif payload_type == "checkpoint_saved":
            self._write_event(
                "💾",
                f"Checkpoint saved at {payload.get('latest_node', 'unknown')}",
                style="yellow",
            )
        elif payload_type == "approval_requested":
            self._set_logo_state("approval")
            self._write_event(
                "⚠️",
                str(payload.get("risk_reason") or "Approval requested"),
                style="yellow",
            )
        elif payload_type == "approval_decision":
            approved = bool(payload.get("approved"))
            self._set_logo_state("tool" if approved else "error")
            reused = " (reused)" if payload.get("reused") else ""
            self._write_event(
                "✅" if approved else "⛔",
                f"Approval {'granted' if approved else 'denied'}"
                f" for attempt {payload.get('attempt', '?')}{reused}",
                style="green" if approved else "red",
            )
        elif payload_type == "final_answer":
            content = payload.get("content") or payload.get("final_answer")
            if content:
                self._set_logo_state("success")
                self._show_final_answer(str(content))

    def _update_plan(self, todos: Any) -> None:
        plan = Text("[Plan] ", style="bold blue")
        if not isinstance(todos, list) or not todos:
            plan.append("No active todos", style="dim")
        else:
            for index, item in enumerate(todos):
                if index:
                    plan.append("  ")
                if not isinstance(item, Mapping):
                    continue
                status = str(item.get("status") or "pending")
                icon = {
                    "completed": "✅",
                    "in_progress": "🔄",
                    "blocked": "⛔",
                    "pending": "⬜",
                }.get(status, "⬜")
                label = item.get("id") or item.get("content") or "todo"
                plan.append(f"{label} {icon}")
        self.query_one("#plan-panel", Static).update(plan)

    def _show_final_answer(self, content: str) -> None:
        if content == self._last_final_answer:
            return
        self._last_final_answer = content
        self._write_event("📝", content, style="bold green")

    def _write_event(self, icon: str, content: str, *, style: str = "") -> None:
        line = Text(f"{icon} ")
        line.append(content, style=style)
        self.query_one("#event-log", RichLog).write(line)

    def _status_text(self, status: str) -> str:
        turn = self._session.get("turn_index", 0)
        return (
            f"session: {self._short_session_id}  ·  turn: {turn}  ·  "
            f"workspace: {self.session_workspace}  ·  {status}"
        )

    def _set_status(self, status: str) -> None:
        self.query_one("#status-bar", Static).update(self._status_text(status))

    def _set_logo_state(self, state: LogoState) -> None:
        self.query_one("#ascii-logo", AuditAgentLogo).set_state(state)

    def _schedule_logo_idle(self) -> None:
        self._cancel_logo_reset()
        self._logo_reset_timer = self.set_timer(
            1.5,
            lambda: self._set_logo_state("idle"),
        )

    def _cancel_logo_reset(self) -> None:
        if self._logo_reset_timer is not None:
            self._logo_reset_timer.stop()
            self._logo_reset_timer = None

    def action_clear_events(self) -> None:
        self.query_one("#event-log", RichLog).clear()

    def on_unmount(self) -> None:
        self._cancel_logo_reset()
        with self._gate_lock:
            gates = list(self._pending_gates.values())
        for gate in gates:
            gate.resolve(False, "Audit Agent TUI closed before approval was decided.")


def _event_target(args: Any) -> str:
    if not isinstance(args, Mapping):
        return _compact_value(args)
    for key in ("file_path", "path", "command", "query", "pattern"):
        value = args.get(key)
        if value not in (None, ""):
            return _compact_value(value, limit=240)
    return _compact_value(args)


def _result_ok(payload: Mapping[str, Any], result: Any) -> bool:
    if isinstance(payload.get("ok"), bool):
        return bool(payload["ok"])
    if isinstance(result, Mapping) and isinstance(result.get("ok"), bool):
        return bool(result["ok"])
    return True


def _compact_value(value: Any, *, limit: int = 800) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def run_tui(
    workspace: str | PathLike[str] | None = None,
    *,
    max_attempts: int = 3,
    approval_mode: str = "inline",
    checkpoint_mode: str = "light",
    trace_mode: str = "on",
) -> None:
    """Create and run the Audit Agent Textual application."""

    AuditAgentTuiApp(
        session_workspace=workspace,
        max_attempts=max_attempts,
        approval_mode=approval_mode,
        checkpoint_mode=checkpoint_mode,
        trace_mode=trace_mode,
    ).run()


if __name__ == "__main__":
    run_tui()
