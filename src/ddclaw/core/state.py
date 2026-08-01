"""Shared runtime state for DDclaw tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path

from ddclaw.core.approval import (
    ApprovalHandler,
    ApprovalTracker,
    normalize_approval_mode,
)
from ddclaw.core.checkpoint import normalize_checkpoint_mode
from ddclaw.core.paths import ensure_workspace, resolve_workspace_path
from ddclaw.core.trace import normalize_trace_mode


@dataclass(frozen=True, slots=True)
class RuntimeState:
    """State shared by all tools during one CLI run."""

    workspace: Path
    checkpoint_mode: str = "light"
    trace_mode: str = "summary"
    trace_id: str | None = None
    approval_mode: str = "inline"
    approval_handler: ApprovalHandler | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    approval_tracker: ApprovalTracker = field(
        default_factory=ApprovalTracker,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        canonical_workspace = ensure_workspace(self.workspace)
        object.__setattr__(self, "workspace", canonical_workspace)
        object.__setattr__(
            self,
            "checkpoint_mode",
            normalize_checkpoint_mode(self.checkpoint_mode),
        )
        object.__setattr__(
            self,
            "trace_mode",
            normalize_trace_mode(self.trace_mode),
        )
        object.__setattr__(
            self,
            "approval_mode",
            normalize_approval_mode(self.approval_mode),
        )

    @classmethod
    def create(
        cls,
        workspace: str | PathLike[str],
        *,
        checkpoint_mode: str | None = None,
        trace_mode: str | None = None,
        trace_id: str | None = None,
        approval_mode: str | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> "RuntimeState":
        """Construct state from a string or path-like workspace value."""

        return cls(
            workspace=Path(workspace),
            checkpoint_mode=normalize_checkpoint_mode(checkpoint_mode),
            trace_mode=normalize_trace_mode(trace_mode),
            trace_id=trace_id,
            approval_mode=normalize_approval_mode(approval_mode),
            approval_handler=approval_handler,
        )

    def resolve_path(
        self,
        requested_path: str | PathLike[str],
        *,
        must_exist: bool = False,
    ) -> Path:
        """Resolve a requested path within this state's workspace."""

        return resolve_workspace_path(
            self.workspace,
            requested_path,
            must_exist=must_exist,
        )


def create_runtime(
    workspace: str | PathLike[str],
    *,
    approval_mode: str | None = "inline",
    approval_handler: ApprovalHandler | None = None,
    checkpoint_mode: str | None = "light",
    resume_from: str | PathLike[str] | None = None,
    trace_mode: str | None = "on",
    trace_id: str | None = None,
) -> RuntimeState:
    """Create one configured runtime, optionally targeting a resume workspace."""

    runtime_workspace = workspace if resume_from is None else resume_from
    return RuntimeState.create(
        runtime_workspace,
        checkpoint_mode=checkpoint_mode,
        trace_mode=trace_mode,
        trace_id=trace_id,
        approval_mode=approval_mode,
        approval_handler=approval_handler,
    )
