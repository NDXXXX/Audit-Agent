"""Thread-safe human approval primitives for the DDclaw TUI."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Lock

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from ddclaw.core.approval import ApprovalDecision, ApprovalRequest


class ApprovalGate:
    """Synchronize one blocking BashTool request with the Textual UI thread."""

    def __init__(self, request: ApprovalRequest, workspace: Path) -> None:
        self.request = request
        self.workspace = workspace.resolve()
        self._ready = Event()
        self._lock = Lock()
        self._decision: ApprovalDecision | None = None

    @property
    def resolved(self) -> bool:
        """Return whether a human or shutdown path supplied a decision."""

        return self._ready.is_set()

    @property
    def decision(self) -> ApprovalDecision | None:
        """Return the current decision without blocking."""

        with self._lock:
            return self._decision

    def resolve(self, approved: bool, reason: str = "") -> bool:
        """Resolve this gate once; return false if it was already resolved."""

        with self._lock:
            if self._decision is not None:
                return False
            self._decision = ApprovalDecision(
                approved=approved,
                reason=reason or (
                    "Approved in the DDclaw TUI."
                    if approved
                    else "Denied in the DDclaw TUI."
                ),
            )
            self._ready.set()
        return True

    def wait(self, timeout: float | None = None) -> ApprovalDecision:
        """Block the Agent thread until the UI resolves this request."""

        if not self._ready.wait(timeout):
            raise TimeoutError(
                f"Approval request {self.request.id} was not resolved in time."
            )
        with self._lock:
            if self._decision is None:  # pragma: no cover - defensive invariant
                raise RuntimeError("Approval gate resolved without a decision.")
            return self._decision


class ApprovalRequestedMessage(Message):
    """Thread-safe message asking the App to display an approval modal."""

    def __init__(self, gate: ApprovalGate) -> None:
        self.gate = gate
        super().__init__()


class ApprovalModal(ModalScreen[bool]):
    """Modal decision screen for one risky BashTool command."""

    BINDINGS = [
        Binding("y", "approve", "Approve", show=False, priority=True),
        Binding("enter", "approve", "Approve", show=False, priority=True),
        Binding("n", "deny", "Deny", show=False, priority=True),
        Binding("escape", "deny", "Deny", show=False, priority=True),
    ]

    CSS = """
    ApprovalModal {
        align: center middle;
        background: $background 65%;
    }

    #approval-dialog {
        width: 92%;
        max-width: 92;
        height: 86%;
        max-height: 32;
        padding: 1 2;
        border: round $warning;
        background: $surface;
    }

    #approval-title {
        width: 100%;
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }

    .approval-label {
        margin-bottom: 1;
    }

    #approval-command {
        width: 100%;
        min-height: 5;
        padding: 1;
        margin-bottom: 1;
        border: solid $primary;
        background: $boost;
    }

    #approval-actions {
        height: 3;
        align-horizontal: right;
    }

    #approval-actions Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        request: ApprovalRequest | ApprovalGate,
        workspace: Path | None = None,
    ) -> None:
        super().__init__()
        if isinstance(request, ApprovalGate):
            self.request = request.request
            self.workspace = request.workspace
        else:
            if workspace is None:
                raise TypeError("workspace is required with an ApprovalRequest")
            self.request = request
            self.workspace = workspace.resolve()

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="approval-dialog"):
            yield Label("⚠ Approval required", id="approval-title")
            yield Label(
                f"Tool: {self.request.tool_name}",
                classes="approval-label",
            )
            yield Label(
                f"Risk: {self.request.risk_reason}",
                classes="approval-label",
            )
            yield Label(
                f"Workspace: {self.workspace}",
                classes="approval-label",
            )
            yield Static(
                self.request.command,
                id="approval-command",
                markup=False,
            )
            with Horizontal(id="approval-actions"):
                yield Button(
                    "[Y] Approve",
                    id="approve",
                    variant="success",
                )
                yield Button(
                    "[N] Deny",
                    id="deny",
                    variant="error",
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "approve":
            self.action_approve()
        elif event.button.id == "deny":
            self.action_deny()

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)
