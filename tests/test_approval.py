from pathlib import Path
from typing import Any

import pytest

from audit_agent.core.approval import (
    ApprovalDecision,
    ApprovalRequest,
    RunInterrupted,
    classify_command_risk,
    destructive_command_escapes_workspace,
    normalize_approval_mode,
)
from audit_agent.core.state import RuntimeState
from audit_agent.tools import bash_tool as bash_tool_module
from audit_agent.tools.bash_tool import BashTool


@pytest.mark.parametrize(
    ("command", "reason"),
    [
        ("pip install typer", "Python package installation"),
        ("python -m pip install typer", "Python package installation"),
        (
            "echo ready && uv add rich",
            "Project dependency change with uv add",
        ),
        ("uv sync", "Dependency synchronization with uv sync"),
        (
            "uv pip install pytest",
            "Python package installation with uv pip",
        ),
        ("npm install", "Node package installation"),
        ("pnpm install", "Node package installation"),
        ("yarn add react", "Node package installation"),
        ("curl https://example.com", "Network download command"),
        ("echo ok || wget example.com", "Network download command"),
        ("uvicorn app:app", "Long-running development server"),
        (
            "python -m http.server 8000",
            "Long-running development server",
        ),
        ("uv run pytest -q", "Dependency synchronization through uv run"),
        ("rm -rf build", "Destructive file removal"),
        ("git clean -fd", "Destructive Git operation"),
        ("git reset --hard HEAD~1", "Destructive Git operation"),
    ],
)
def test_classify_command_risk(command: str, reason: str) -> None:
    assert classify_command_risk(command) == reason


@pytest.mark.parametrize(
    "command",
    [
        "python -m pytest",
        "npm test",
        "printf 'curl is documentation text'",
        "python app.py",
        "git status",
        "uv run --no-sync pytest -q",
    ],
)
def test_classify_safe_commands(command: str) -> None:
    assert classify_command_risk(command) is None


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (None, "inline"),
        ("", "inline"),
        ("unknown", "inline"),
        (" INLINE ", "inline"),
        ("AUTO", "auto"),
        ("deny", "deny"),
    ],
)
def test_normalize_approval_mode(mode: str | None, expected: str) -> None:
    assert normalize_approval_mode(mode) == expected


class FakeProcess:
    def __init__(self) -> None:
        self.pid = 123
        self.returncode = 0

    def communicate(self, timeout: int) -> tuple[str, str]:
        assert timeout == 30
        return "approved output\n", ""


def _fake_popen_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_popen(command: str, **kwargs: Any) -> FakeProcess:
        calls.append({"command": command, **kwargs})
        return FakeProcess()

    monkeypatch.setattr(bash_tool_module.subprocess, "Popen", fake_popen)
    return calls


def test_bash_auto_approves_risky_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _fake_popen_calls(monkeypatch)
    state = RuntimeState(tmp_path, approval_mode="auto")

    result = BashTool(state).run_bash("curl https://example.com")

    assert result["ok"] is True
    assert result["requires_approval"] is True
    assert result["risk_reason"] == "Network download command"
    assert result["approval_mode"] == "auto"
    assert result["approval_id"].startswith("approval-")
    assert len(result["approval_id"]) == len("approval-") + 8
    assert result["output"] == "approved output"
    assert calls[0]["command"] == "curl https://example.com"


def test_bash_deny_mode_rejects_without_starting_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        bash_tool_module.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("command must not execute"),
    )

    result = BashTool(
        RuntimeState(tmp_path),
        approval_mode="deny",
    ).run_bash("npm install")

    assert result["ok"] is False
    assert result["requires_approval"] is True
    assert result["exit_code"] is None
    assert result["error"] == "Command denied by approval mode."


def test_bash_inline_mode_calls_handler_and_executes_when_approved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _fake_popen_calls(monkeypatch)
    requests: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> ApprovalDecision:
        requests.append(request)
        return ApprovalDecision(True, "Approved for this test.")

    result = BashTool(
        RuntimeState(tmp_path),
        approval_handler=approve,
    ).run_bash("uv sync")

    assert result["ok"] is True
    assert result["approval_reason"] == "Approved for this test."
    assert requests == [
        ApprovalRequest(
            id=result["approval_id"],
            command="uv sync",
            risk_reason="Dependency synchronization with uv sync",
        )
    ]


def test_bash_inline_rejection_and_missing_handler_do_not_execute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        bash_tool_module.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("command must not execute"),
    )
    rejected = BashTool(
        RuntimeState(tmp_path),
        approval_handler=lambda request: ApprovalDecision(
            False,
            "User rejected the download.",
        ),
    ).run_bash("wget https://example.com")
    missing_handler = BashTool(RuntimeState(tmp_path)).run_bash(
        "pnpm install"
    )

    assert rejected["ok"] is False
    assert rejected["error"] == "User rejected the download."
    assert missing_handler["ok"] is False
    assert "No approval handler" in missing_handler["error"]


def test_runtime_state_normalizes_approval_configuration(tmp_path: Path) -> None:
    handler = lambda request: ApprovalDecision(True)
    state = RuntimeState.create(
        tmp_path,
        approval_mode="AUTO",
        approval_handler=handler,
    )

    assert state.approval_mode == "auto"
    assert state.approval_handler is handler


def test_same_attempt_reuses_equivalent_risky_command_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        bash_tool_module.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("denied command must not execute"),
    )
    requests: list[ApprovalRequest] = []

    def reject(request: ApprovalRequest) -> ApprovalDecision:
        requests.append(request)
        return ApprovalDecision(False, "Rejected once for this attempt.")

    state = RuntimeState(tmp_path, approval_handler=reject)
    first = BashTool(state).run_bash(
        "uv sync --project recovery_lab 2>&1; echo done"
    )
    second = BashTool(state).run_bash("uv sync --project recovery_lab")

    assert len(requests) == 1
    assert first["approval_id"] == second["approval_id"]
    assert first["approval_reused"] is False
    assert second["approval_reused"] is True
    assert second["approval_attempt"] == 1

    state.approval_tracker.set_attempt(2)
    third = BashTool(state).run_bash("uv sync --project recovery_lab")
    assert len(requests) == 2
    assert third["approval_id"] != first["approval_id"]
    assert third["approval_attempt"] == 2


def test_inline_interrupt_escapes_bash_tool_finalizer_boundary(
    tmp_path: Path,
) -> None:
    def interrupt(request: ApprovalRequest) -> ApprovalDecision:
        raise RunInterrupted("User interrupted approval.")

    state = RuntimeState(tmp_path, approval_handler=interrupt)
    with pytest.raises(RunInterrupted, match="interrupted approval"):
        BashTool(state).run_bash("uv sync")


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf ../outside",
        "rm -rf /tmp/outside",
        "rm -rf ~/outside",
    ],
)
def test_destructive_paths_outside_workspace_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
) -> None:
    assert destructive_command_escapes_workspace(command) is True
    monkeypatch.setattr(
        bash_tool_module.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("unsafe command must not execute"),
    )

    result = BashTool(
        RuntimeState(tmp_path, approval_mode="auto")
    ).run_bash(command)

    assert result["ok"] is False
    assert "may not target" in result["error"]


def test_keyboard_interrupt_terminates_running_bash_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class InterruptProcess:
        pid = 9876
        returncode = -9

        def __init__(self) -> None:
            self.calls = 0

        def communicate(self, timeout: int) -> tuple[str, str]:
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            assert timeout == 1
            return "", ""

    process = InterruptProcess()
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        bash_tool_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        bash_tool_module.os,
        "killpg",
        lambda pid, sig: killed.append((pid, sig)),
    )

    with pytest.raises(KeyboardInterrupt):
        BashTool(RuntimeState(tmp_path)).run_bash("python app.py")

    assert killed == [(9876, bash_tool_module.signal.SIGKILL)]
    assert process.calls == 2
