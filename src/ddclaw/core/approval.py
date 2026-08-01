"""Command-risk classification and human approval primitives."""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Any

RISK_PATTERNS = [
    (
        r"(?:^|&&|\|\||;)\s*(?:python\s+-m\s+)?pip\s+install\b",
        "Python package installation",
    ),
    (
        r"(?:^|&&|\|\||;)\s*uv\s+add\b",
        "Project dependency change with uv add",
    ),
    (
        r"(?:^|&&|\|\||;)\s*uv\s+sync\b",
        "Dependency synchronization with uv sync",
    ),
    (
        r"(?:^|&&|\|\||;)\s*uv\s+pip\s+install\b",
        "Python package installation with uv pip",
    ),
    (
        r"(?:^|&&|\|\||;)\s*npm\s+install\b",
        "Node package installation",
    ),
    (
        r"(?:^|&&|\|\||;)\s*pnpm\s+install\b",
        "Node package installation",
    ),
    (
        r"(?:^|&&|\|\||;)\s*yarn\s+(?:install\b|add\b)",
        "Node package installation",
    ),
    (
        r"(?:^|&&|\|\||;)\s*(?:curl|wget)\b",
        "Network download command",
    ),
    (
        r"(?:^|&&|\|\||;)\s*uvicorn\b",
        "Long-running development server",
    ),
    (
        r"(?:^|&&|\|\||;)\s*python\s+-m\s+http\.server\b",
        "Long-running development server",
    ),
]

VALID_APPROVAL_MODES = {"inline", "auto", "deny"}


class RunInterrupted(KeyboardInterrupt):
    """A user-requested run interruption that must reach the run finalizer."""


@dataclass(frozen=True)
class ApprovalRecord:
    """One cached human/policy decision within a graph attempt."""

    request: "ApprovalRequest"
    decision: "ApprovalDecision"


class ApprovalTracker:
    """Cache approval decisions per attempt and queue traceable events."""

    def __init__(self) -> None:
        self._attempt = 1
        self._records: dict[tuple[int, str, str], ApprovalRecord] = {}
        self._events: list[dict[str, Any]] = []
        self._lock = Lock()

    @property
    def attempt(self) -> int:
        with self._lock:
            return self._attempt

    def set_attempt(self, attempt: int) -> None:
        normalized = max(1, int(attempt))
        with self._lock:
            self._attempt = normalized

    def lookup(self, *, tool_name: str, command: str) -> ApprovalRecord | None:
        with self._lock:
            return self._records.get(
                (self._attempt, tool_name, _normalize_command(command))
            )

    def record_request(self, request: "ApprovalRequest") -> None:
        with self._lock:
            self._events.append(
                {
                    "type": "approval_requested",
                    "approval_id": request.id,
                    "attempt": self._attempt,
                    "tool_name": request.tool_name,
                    "command": request.command,
                    "risk_reason": request.risk_reason,
                }
            )

    def record_decision(
        self,
        request: "ApprovalRequest",
        decision: "ApprovalDecision",
        *,
        reused: bool,
    ) -> None:
        with self._lock:
            if not reused:
                self._records[
                    (
                        self._attempt,
                        request.tool_name,
                        _normalize_command(request.command),
                    )
                ] = ApprovalRecord(request=request, decision=decision)
            self._events.append(
                {
                    "type": "approval_decision",
                    "approval_id": request.id,
                    "attempt": self._attempt,
                    "tool_name": request.tool_name,
                    "command": request.command,
                    "risk_reason": request.risk_reason,
                    "approved": decision.approved,
                    "reason": decision.reason,
                    "reused": reused,
                }
            )

    def drain_events(self) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events


@dataclass(frozen=True)
class ApprovalRequest:
    """One risky command awaiting an approval policy decision."""

    id: str
    command: str
    risk_reason: str
    tool_name: str = "BashTool"


@dataclass(frozen=True)
class ApprovalDecision:
    """Human or policy response to an approval request."""

    approved: bool
    reason: str = ""


ApprovalHandler = Callable[[ApprovalRequest], ApprovalDecision]


def classify_command_risk(command: str) -> str | None:
    """Return the first matching risk reason, or ``None`` when safe."""

    for pattern, reason in RISK_PATTERNS:
        if re.search(pattern, command):
            return reason
    for segment in _command_segments(command):
        tokens = _shell_tokens(segment)
        if _is_implicit_uv_run_sync(tokens):
            return "Dependency synchronization through uv run"
        if _is_destructive_remove(tokens):
            return "Destructive file removal"
        if _is_destructive_git(tokens):
            return "Destructive Git operation"
    return None


def destructive_command_escapes_workspace(command: str) -> bool:
    """Return whether a destructive command names an outside-style path."""

    for segment in _command_segments(command):
        tokens = _shell_tokens(segment)
        if not _is_destructive_remove(tokens):
            continue
        for token in tokens[1:]:
            if token.startswith("-"):
                continue
            if token.startswith(("/", "~")):
                return True
            if ".." in token.replace("\\", "/").split("/"):
                return True
    return False


def normalize_approval_mode(mode: str | None) -> str:
    """Normalize a configured mode, falling back to interactive approval."""

    normalized = mode.strip().lower() if isinstance(mode, str) else ""
    return normalized if normalized in VALID_APPROVAL_MODES else "inline"


def _normalize_command(command: str) -> str:
    """Build a stable approval key for the risky operation in a shell command.

    Agents commonly append redirections or an ``echo`` command while retrying.
    Those presentation-only differences must not create a second prompt in the
    same graph attempt.
    """

    for segment in _command_segments(command):
        tokens = _shell_tokens(segment)
        operation = _risky_operation_tokens(tokens)
        if operation:
            return " ".join(operation)
    return " ".join(command.split())


def _command_segments(command: str) -> list[str]:
    return [
        segment.strip()
        for segment in re.split(r"&&|\|\||;", command)
        if segment.strip()
    ]


def _shell_tokens(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _risky_operation_tokens(tokens: list[str]) -> list[str]:
    """Return the executable portion used to deduplicate approval prompts."""

    if not tokens:
        return []
    starts: list[tuple[str, ...]] = [
        ("uv", "sync"),
        ("uv", "add"),
        ("uv", "pip", "install"),
        ("uv", "run"),
        ("pip", "install"),
        ("npm", "install"),
        ("pnpm", "install"),
        ("yarn", "install"),
        ("yarn", "add"),
        ("curl",),
        ("wget",),
        ("uvicorn",),
        ("rm",),
        ("git", "clean"),
        ("git", "reset"),
    ]
    for index in range(len(tokens)):
        tail = tokens[index:]
        for prefix in starts:
            if tuple(tail[: len(prefix)]) != prefix:
                continue
            if prefix == ("uv", "run") and "--no-sync" in tail:
                continue
            cleaned: list[str] = []
            for token in tail:
                if re.fullmatch(r"(?:\d*)?>&?\d*", token) or token in {
                    ">",
                    ">>",
                    "<",
                    "2>",
                    "2>>",
                }:
                    break
                cleaned.append(token)
            return cleaned
    return []


def _is_implicit_uv_run_sync(tokens: list[str]) -> bool:
    try:
        uv_index = tokens.index("uv")
    except ValueError:
        return False
    following = tokens[uv_index + 1 :]
    return bool(following) and following[0] == "run" and "--no-sync" not in following


def _is_destructive_remove(tokens: list[str]) -> bool:
    if not tokens or tokens[0] != "rm":
        return False
    options = [token for token in tokens[1:] if token.startswith("-")]
    recursive = any(
        token == "--recursive" or (not token.startswith("--") and "r" in token)
        for token in options
    )
    force = any(
        token == "--force" or (not token.startswith("--") and "f" in token)
        for token in options
    )
    return recursive or force


def _is_destructive_git(tokens: list[str]) -> bool:
    if len(tokens) < 2 or tokens[0] != "git":
        return False
    if tokens[1] == "clean":
        return True
    return tokens[1] == "reset" and "--hard" in tokens[2:]
