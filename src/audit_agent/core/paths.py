"""Audit Agent helpers for safely resolving paths inside a workspace."""

from __future__ import annotations

from os import PathLike
from pathlib import Path


class WorkspacePathError(ValueError):
    """Raised when a requested path escapes the configured workspace."""


def ensure_workspace(workspace: str | PathLike[str]) -> Path:
    """Create *workspace* when needed and return its canonical path."""

    path = Path(workspace).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"Workspace is not a directory: {resolved}")
    return resolved


def resolve_workspace_path(
    workspace: str | PathLike[str],
    requested_path: str | PathLike[str],
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve a path and reject anything outside *workspace*.

    Absolute paths are accepted only when they resolve inside the workspace.
    ``Path.resolve`` also resolves existing symbolic links, preventing a link
    inside the workspace from being used to reach a file outside it.
    """

    root = Path(workspace).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Workspace is not a directory: {root}")

    requested = Path(requested_path).expanduser()
    candidate = requested if requested.is_absolute() else root / requested

    try:
        resolved = candidate.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Path does not exist: {requested_path}") from exc

    if not resolved.is_relative_to(root):
        raise WorkspacePathError(
            f"Path escapes workspace: {requested_path!s} (workspace: {root})"
        )
    return resolved


def workspace_relative_path(workspace: Path, path: Path) -> str:
    """Return a stable POSIX-style path relative to *workspace*."""

    return path.resolve(strict=False).relative_to(workspace).as_posix()
