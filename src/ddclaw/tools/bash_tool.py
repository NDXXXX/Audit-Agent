"""Shell-command execution rooted in a configured workspace."""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from ddclaw.core.approval import (
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    classify_command_risk,
    destructive_command_escapes_workspace,
    normalize_approval_mode,
)
from ddclaw.core.state import RuntimeState


class BashInput(BaseModel):
    """Arguments accepted by the bash tool."""

    command: str = Field(description="Shell command to execute in the workspace.")
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        description="Maximum execution time in seconds.",
    )


class BashTool:
    """Execute a shell command with the workspace as its current directory."""

    def __init__(
        self,
        state: RuntimeState,
        *,
        approval_mode: str | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        self.state = state
        self.approval_mode = normalize_approval_mode(
            state.approval_mode if approval_mode is None else approval_mode
        )
        self.approval_handler = (
            state.approval_handler
            if approval_handler is None
            else approval_handler
        )

    def __call__(
        self,
        command: str,
        timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        return self.run_bash(
            command=command,
            timeout_seconds=timeout_seconds,
        )

    def run(
        self,
        command: str,
        timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        """Compatibility alias used by the structured-tool registry."""

        return self.run_bash(
            command=command,
            timeout_seconds=timeout_seconds,
        )

    def run_bash(
        self,
        command: str,
        timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        """Classify, approve when needed, and execute one shell command."""

        if not command.strip():
            raise ValueError("command must not be empty")
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be greater than or equal to 1")

        risk_reason = classify_command_risk(command)
        approval_metadata: dict[str, Any] = {
            "requires_approval": risk_reason is not None,
        }
        if risk_reason is not None:
            if destructive_command_escapes_workspace(command):
                return {
                    "ok": False,
                    "command": command,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "output": "",
                    "requires_approval": True,
                    "risk_reason": risk_reason,
                    "approval_mode": self.approval_mode,
                    "error": (
                        "Destructive commands may not target absolute, home, "
                        "or parent-directory paths."
                    ),
                }
            request = ApprovalRequest(
                id=f"approval-{uuid4().hex[:8]}",
                command=command,
                risk_reason=risk_reason,
            )
            approval_metadata.update(
                {
                    "approval_id": request.id,
                    "risk_reason": risk_reason,
                    "approval_mode": self.approval_mode,
                }
            )
            cached = self.state.approval_tracker.lookup(
                tool_name=request.tool_name,
                command=request.command,
            )
            if cached is None:
                self.state.approval_tracker.record_request(request)
                decision = self._approval_decision(request)
                reused = False
            else:
                request = cached.request
                decision = cached.decision
                reused = True
                approval_metadata["approval_id"] = request.id
            self.state.approval_tracker.record_decision(
                request,
                decision,
                reused=reused,
            )
            approval_metadata["approval_reused"] = reused
            approval_metadata["approval_attempt"] = (
                self.state.approval_tracker.attempt
            )
            approval_metadata["approval_decision"] = (
                "approved" if decision.approved else "denied"
            )
            if not decision.approved:
                reason = decision.reason or "Command was not approved."
                return {
                    "ok": False,
                    "command": command,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "output": "",
                    **approval_metadata,
                    "error": reason,
                }
            if decision.reason:
                approval_metadata["approval_reason"] = decision.reason

        process = subprocess.Popen(
            command,
            cwd=self.state.workspace,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=os.name == "posix",
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except (KeyboardInterrupt, GeneratorExit):
            _terminate_process(process)
            try:
                process.communicate(timeout=1)
            except (OSError, subprocess.SubprocessError):
                pass
            raise
        except subprocess.TimeoutExpired as exc:
            _terminate_process(process)
            stdout, stderr = process.communicate()
            partial_output = _format_command_output(
                stdout=stdout,
                stderr=stderr,
                return_code=process.returncode,
            )
            suffix = f"\nPartial output:\n{partial_output}" if partial_output else ""
            raise TimeoutError(
                f"Command timed out after {timeout_seconds} seconds{suffix}"
            ) from exc

        output = _format_command_output(
            stdout=stdout,
            stderr=stderr,
            return_code=process.returncode,
        )
        return {
            "ok": process.returncode == 0,
            "command": command,
            "exit_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "output": output or "Command completed successfully with no output.",
            **approval_metadata,
        }

    def _approval_decision(
        self,
        request: ApprovalRequest,
    ) -> ApprovalDecision:
        if self.approval_mode == "auto":
            return ApprovalDecision(
                approved=True,
                reason="Automatically approved by approval mode.",
            )
        if self.approval_mode == "deny":
            return ApprovalDecision(
                approved=False,
                reason="Command denied by approval mode.",
            )
        if self.approval_handler is None:
            return ApprovalDecision(
                approved=False,
                reason="No approval handler is configured for inline mode.",
            )

        try:
            decision = self.approval_handler(request)
        except Exception as exc:
            return ApprovalDecision(
                approved=False,
                reason=f"Approval handler failed: {type(exc).__name__}: {exc}",
            )
        if not isinstance(decision, ApprovalDecision):
            return ApprovalDecision(
                approved=False,
                reason="Approval handler returned an invalid decision.",
            )
        return decision


def _format_command_output(
    *,
    stdout: str,
    stderr: str,
    return_code: int | None,
) -> str:
    sections: list[str] = []
    if stdout:
        sections.append(stdout.rstrip())
    if stderr:
        sections.append(f"STDERR:\n{stderr.rstrip()}")
    if return_code:
        sections.append(f"Command exited with code {return_code}.")
    return "\n".join(sections)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    """Best-effort termination for a command running in its own session."""

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        pass
