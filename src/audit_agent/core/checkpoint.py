"""Workspace checkpoint persistence and recovery for Audit Agent runs."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import socket
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    messages_from_dict,
    messages_to_dict,
)

if TYPE_CHECKING:
    from audit_agent.core.state import RuntimeState

VALID_CHECKPOINT_MODES = {"light", "strict", "off"}
_CHECKPOINT_DIRECTORY = ".audit/checkpoints"
_CHECKPOINT_PATH = "checkpoint.json"
_STRICT_STATE_PATH = "state.json"
_STRICT_EVENTS_PATH = "events.jsonl"
_RECOVERY_PATH = "RECOVERY.md"
_SNAPSHOT_GIT_DIRECTORY = "workspace.git"
_RUN_LEASE_PATH = "run.json"
_TERMINAL_STATUSES = {"finished", "interrupted", "failed"}
_REGENERATED_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}


def normalize_checkpoint_mode(mode: str | None) -> str:
    """Normalize a checkpoint mode, defaulting invalid values to ``light``."""

    normalized = mode.strip().lower() if isinstance(mode, str) else ""
    return normalized if normalized in VALID_CHECKPOINT_MODES else "light"


class CheckpointManager:
    """Save and restore graph state together with workspace Git snapshots."""

    def __init__(self, runtime: RuntimeState, task: str = "") -> None:
        self.runtime = runtime
        self.workspace = runtime.workspace
        self.mode = normalize_checkpoint_mode(runtime.checkpoint_mode)
        self.task = task
        self.root = self.workspace / _CHECKPOINT_DIRECTORY

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def begin_run(self, run_id: str) -> None:
        """Acquire a lightweight workspace run lease."""

        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        lease_path = self.root / _RUN_LEASE_PATH
        if lease_path.is_file():
            lease = _read_json_object(lease_path)
            if _lease_is_active(lease) and lease.get("run_id") != run_id:
                raise RuntimeError(
                    "A Audit Agent run is already active in this workspace "
                    f"(pid={lease.get('pid')}, run_id={lease.get('run_id')})."
                )
        now = _utc_now()
        _write_json(
            lease_path,
            {
                "run_id": run_id,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "status": "running",
                "started_at": now,
                "updated_at": now,
            },
        )

    def save(
        self,
        state: Mapping[str, Any],
        *,
        status: str = "running",
        latest_node: str | None = None,
        event: Any = None,
        snapshot: bool = True,
    ) -> dict[str, Any] | None:
        """Persist state, a workspace snapshot, and recovery instructions."""

        if not self.enabled:
            return None

        self.root.mkdir(parents=True, exist_ok=True)
        saved_at = _utc_now()
        task = str(state.get("task") or self.task)

        if self.mode == "strict":
            strict_state = _serialize_state(state)
            _write_json(self.root / _STRICT_STATE_PATH, strict_state)
            if event is not None:
                event_record = {
                    "saved_at": saved_at,
                    "status": status,
                    "latest_node": latest_node,
                    "event": _jsonable(event),
                }
                _append_json_line(
                    self.root / _STRICT_EVENTS_PATH,
                    event_record,
                )

        checkpoint_path = self.root / _CHECKPOINT_PATH
        previous = _read_json_if_present(checkpoint_path)
        if snapshot:
            manifest = workspace_manifest(self.workspace)
            git_commit = snapshot_workspace_git(
                self.workspace,
                self.root,
                message=f"Audit Agent checkpoint: {latest_node or status}",
            )
        else:
            manifest = previous.get("workspace_manifest", [])
            git_commit = previous.get("git_commit")
        payload = {
            "version": 1,
            "saved_at": saved_at,
            "task": task,
            "status": status,
            "latest_node": latest_node,
            "checkpoint_mode": self.mode,
            "workspace": str(self.workspace),
            "workspace_manifest": manifest,
            "git_commit": git_commit,
            "resume_command": resume_command(self.workspace),
            "state_summary": _state_summary(state),
        }
        _write_json(checkpoint_path, payload)
        _write_text(
            self.root / _RECOVERY_PATH,
            build_recovery_markdown(payload),
        )
        self._update_lease(status)
        return checkpoint_saved_event(payload, checkpoint_path=checkpoint_path)

    def finalize(
        self,
        state: Mapping[str, Any],
        *,
        status: str,
        latest_node: str | None,
        event: Any = None,
    ) -> dict[str, Any] | None:
        """Atomically mark a run terminal without a slow workspace snapshot."""

        if not self.enabled:
            return None
        if status not in _TERMINAL_STATUSES:
            raise ValueError(f"Unsupported terminal checkpoint status: {status}")

        self.root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.root / _CHECKPOINT_PATH
        previous = _read_json_if_present(checkpoint_path)
        saved_at = _utc_now()
        task = str(state.get("task") or previous.get("task") or self.task)
        duplicate_terminal = (
            previous.get("status") == status
            and previous.get("latest_node") == latest_node
        )
        payload = {
            **previous,
            "version": 1,
            "saved_at": saved_at,
            "task": task,
            "status": status,
            "latest_node": latest_node,
            "checkpoint_mode": self.mode,
            "workspace": str(self.workspace),
            "resume_command": resume_command(self.workspace),
        }
        payload.setdefault("workspace_manifest", [])
        payload.setdefault("git_commit", None)
        payload.setdefault("state_summary", {})

        # The terminal marker is intentionally first and atomic. If a second
        # signal arrives, recovery still sees the correct run status.
        _write_json(checkpoint_path, payload)

        if self.mode == "strict":
            _write_json(self.root / _STRICT_STATE_PATH, _serialize_state(state))
            if not duplicate_terminal:
                _append_json_line(
                    self.root / _STRICT_EVENTS_PATH,
                    {
                        "saved_at": saved_at,
                        "status": status,
                        "latest_node": latest_node,
                        "event": _jsonable(
                            event
                            or {
                                "type": "run_terminal",
                                "status": status,
                            }
                        ),
                    },
                )

        payload["state_summary"] = _state_summary(state)
        _write_json(checkpoint_path, payload)
        _write_text(
            self.root / _RECOVERY_PATH,
            build_recovery_markdown(payload),
        )
        self._update_lease(status)
        return checkpoint_saved_event(payload, checkpoint_path=checkpoint_path)

    @classmethod
    def load_resume_inputs(
        cls,
        runtime: RuntimeState,
        task: str | None = None,
        max_attempts: int = 3,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Restore a saved workspace and rebuild graph input state."""

        if max_attempts < 1:
            raise ValueError("max_attempts must be greater than or equal to 1")

        manager = cls(runtime, task=task or "")
        checkpoint_path = manager.root / _CHECKPOINT_PATH
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {checkpoint_path}"
            )

        payload = _read_json_object(checkpoint_path)
        stale_running = False
        if payload.get("status") == "running":
            lease = _read_json_if_present(manager.root / _RUN_LEASE_PATH)
            if _lease_is_active(lease):
                raise RuntimeError(
                    "Cannot resume because the checkpoint workspace still has "
                    f"an active run (pid={lease.get('pid')})."
                )
            stale_running = True
            payload["status"] = "interrupted"
            payload["interruption_reason"] = (
                "Recovered a stale running checkpoint with no active run lease."
            )
            payload["saved_at"] = _utc_now()
            _write_json(checkpoint_path, payload)
            _write_text(
                manager.root / _RECOVERY_PATH,
                build_recovery_markdown(payload),
            )
        git_commit = payload.get("git_commit")
        restored_workspace = False
        if isinstance(git_commit, str) and git_commit:
            restore_workspace_git(
                manager.workspace,
                manager.root,
                git_commit,
            )
            restored_workspace = True

        saved_mode = normalize_checkpoint_mode(payload.get("checkpoint_mode"))
        strict_state_path = manager.root / _STRICT_STATE_PATH
        if saved_mode == "strict" and strict_state_path.is_file():
            inputs = _deserialize_state(
                _read_json_object(strict_state_path),
                runtime,
            )
        else:
            raw_summary = payload.get("state_summary", {})
            inputs = (
                dict(raw_summary)
                if isinstance(raw_summary, Mapping)
                else {}
            )
            summary = str(
                inputs.get("context_summary")
                or inputs.get("history_summary")
                or ""
            ).strip()
            inputs["messages"] = (
                [AIMessage(content=f"Recovered context:\n{summary}")]
                if summary
                else []
            )

        inputs["task"] = str(task or payload.get("task") or "")
        inputs["runtime"] = runtime
        inputs["max_attempts"] = max_attempts
        inputs.setdefault("messages", [])
        inputs.setdefault("todos", [])
        inputs.setdefault("research_notes", "")
        inputs.setdefault("sources", [])
        inputs.setdefault("agent_handoffs", [])
        inputs.setdefault("attempts", 0)
        inputs.setdefault("passed", False)

        resume_event = {
            "type": "checkpoint_resumed",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_mode": saved_mode,
            "status": payload.get("status", "running"),
            "latest_node": payload.get("latest_node"),
            "git_commit": git_commit,
            "restored_workspace": restored_workspace,
            "stale_running": stale_running,
            "resume_strategy": "replan_from_saved_state",
        }
        return inputs, resume_event

    def _update_lease(self, status: str) -> None:
        if not self.enabled:
            return
        lease_path = self.root / _RUN_LEASE_PATH
        lease = _read_json_if_present(lease_path)
        if not lease:
            return
        lease["status"] = status
        lease["updated_at"] = _utc_now()
        _write_json(lease_path, lease)


def workspace_manifest(workspace: Path) -> list[dict[str, Any]]:
    """Return a stable content manifest excluding VCS/checkpoint internals."""

    root = workspace.resolve(strict=True)
    manifest: list[dict[str, Any]] = []
    for directory, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(directory)
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            path = current / name
            relative = path.relative_to(root)
            if _is_internal_path(relative):
                continue
            if path.is_symlink():
                manifest.append(
                    {
                        "path": relative.as_posix(),
                        "type": "symlink",
                        "target": str(path.readlink()),
                    }
                )
                continue
            retained_directories.append(name)
        directory_names[:] = retained_directories

        for name in sorted(file_names):
            path = current / name
            relative = path.relative_to(root)
            if _is_internal_path(relative):
                continue
            if path.is_symlink():
                manifest.append(
                    {
                        "path": relative.as_posix(),
                        "type": "symlink",
                        "target": str(path.readlink()),
                    }
                )
                continue
            try:
                size = path.stat().st_size
                digest = _file_sha256(path)
            except OSError as exc:
                manifest.append(
                    {
                        "path": relative.as_posix(),
                        "type": "file",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            manifest.append(
                {
                    "path": relative.as_posix(),
                    "type": "file",
                    "size": size,
                    "sha256": digest,
                }
            )
    return sorted(manifest, key=lambda item: str(item.get("path", "")))


def snapshot_workspace_git(
    workspace: Path,
    checkpoint_root: Path,
    *,
    message: str = "Audit Agent checkpoint",
) -> str | None:
    """Commit workspace contents into an isolated checkpoint Git repository."""

    git_dir = checkpoint_root / _SNAPSHOT_GIT_DIRECTORY
    try:
        if not git_dir.is_dir():
            _run_git(["git", "init", "--bare", str(git_dir)])

        command = _git_command(git_dir, workspace)
        _run_git([*command, "config", "user.name", "Audit Agent Checkpoint"])
        _run_git(
            [*command, "config", "user.email", "checkpoint@audit.local"]
        )
        _run_git([*command, "config", "commit.gpgsign", "false"])
        _run_git(
            [
                *command,
                "add",
                "-A",
                "--",
                ".",
                ":(exclude).audit",
                ":(exclude).git",
                *_snapshot_exclusion_pathspecs(),
            ],
            cwd=workspace,
        )
        _drop_regenerated_snapshot_paths(command)

        current_commit = _git_output(
            [*command, "rev-parse", "--verify", "HEAD"],
            check=False,
        )
        staged_changes = subprocess.run(
            [*command, "diff", "--cached", "--quiet"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).returncode != 0
        if current_commit and not staged_changes:
            return current_commit

        _run_git(
            [
                *command,
                "commit",
                "--allow-empty",
                "--no-gpg-sign",
                "-m",
                message,
            ],
            cwd=workspace,
        )
        return _git_output([*command, "rev-parse", "HEAD"])
    except (OSError, subprocess.SubprocessError):
        return None


def restore_workspace_git(
    workspace: Path,
    checkpoint_root: Path,
    commit: str,
) -> None:
    """Restore tracked workspace files from the isolated snapshot repository."""

    git_dir = checkpoint_root / _SNAPSHOT_GIT_DIRECTORY
    if not git_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint Git repository is missing: {git_dir}")
    command = _git_command(git_dir, workspace)
    _run_git([*command, "reset", "--hard", commit], cwd=workspace)


def resume_command(workspace: Path) -> str:
    """Return the CLI command used to resume a workspace checkpoint."""

    return f"audit --resume {shlex.quote(str(workspace.resolve()))}"


def build_recovery_markdown(payload: Mapping[str, Any]) -> str:
    """Build the human-readable ``RECOVERY.md`` checkpoint guide."""

    manifest = payload.get("workspace_manifest", [])
    file_lines: list[str] = []
    if isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes)):
        for item in manifest:
            if not isinstance(item, Mapping):
                continue
            path = str(item.get("path", ""))
            size = item.get("size")
            detail = f" — {size} bytes" if isinstance(size, int) else ""
            file_lines.append(f"- `{path}`{detail}")
    if not file_lines:
        file_lines.append("- No workspace files were recorded.")

    commit = payload.get("git_commit") or "Unavailable"
    return "\n".join(
        [
            "# Audit Agent Recovery",
            "",
            f"- Task: {payload.get('task', '')}",
            f"- Status: {payload.get('status', 'running')}",
            f"- Latest node: {payload.get('latest_node') or 'unknown'}",
            f"- Checkpoint mode: {payload.get('checkpoint_mode', 'light')}",
            f"- Git commit: `{commit}`",
            "",
            "## Workspace files",
            "",
            *file_lines,
            "",
            "## Resume",
            "",
            "```bash",
            str(payload.get("resume_command", "")),
            "```",
            "",
        ]
    )


def checkpoint_saved_event(
    payload: Mapping[str, Any],
    *,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Build the custom event returned after a successful checkpoint save."""

    return {
        "type": "checkpoint_saved",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_mode": payload.get("checkpoint_mode", "light"),
        "status": payload.get("status", "running"),
        "latest_node": payload.get("latest_node"),
        "git_commit": payload.get("git_commit"),
    }


def _state_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _jsonable(value)
        for key, value in state.items()
        if key not in {"runtime", "messages", "memory_snapshot"}
    }


def _serialize_state(state: Mapping[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in state.items():
        if key == "runtime":
            continue
        if key == "messages" and isinstance(value, Sequence):
            serialized[key] = messages_to_dict(list(value))
        else:
            serialized[key] = _jsonable(value)
    return serialized


def _deserialize_state(
    serialized: Mapping[str, Any],
    runtime: RuntimeState,
) -> dict[str, Any]:
    state = dict(serialized)
    raw_messages = state.get("messages", [])
    if isinstance(raw_messages, list):
        try:
            state["messages"] = messages_from_dict(raw_messages)
        except (KeyError, TypeError, ValueError):
            state["messages"] = []
    else:
        state["messages"] = []
    state["runtime"] = runtime
    return state


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseMessage):
        return messages_to_dict([value])[0]
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item) for item in value]
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _append_json_line(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, default=str))
        stream.write("\n")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Checkpoint JSON is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint JSON must contain an object: {path}")
    return payload


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return _read_json_object(path)
    except (OSError, UnicodeError, ValueError):
        return {}


def _write_text(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_internal_path(relative: Path) -> bool:
    return bool(relative.parts) and (
        relative.parts[0] in {".audit", ".git"}
        or any(part in _REGENERATED_PARTS for part in relative.parts)
        or relative.suffix == ".pyc"
    )


def _snapshot_exclusion_pathspecs() -> list[str]:
    pathspecs = [":(exclude,glob)**/*.pyc"]
    for part in sorted(_REGENERATED_PARTS):
        pathspecs.extend(
            [
                f":(exclude,glob){part}/**",
                f":(exclude,glob)**/{part}/**",
            ]
        )
    return pathspecs


def _drop_regenerated_snapshot_paths(command: list[str]) -> None:
    """Remove generated paths that an older checkpoint index may track."""

    tracked = _git_output([*command, "ls-files", "-z"], check=False)
    ignored = [
        path
        for path in tracked.split("\0")
        if path and _is_internal_path(Path(path))
    ]
    for index in range(0, len(ignored), 200):
        _run_git(
            [
                *command,
                "update-index",
                "--force-remove",
                "--",
                *ignored[index : index + 200],
            ]
        )


def _lease_is_active(lease: Mapping[str, Any]) -> bool:
    if lease.get("status") != "running":
        return False
    if lease.get("hostname") != socket.gethostname():
        return False
    pid = lease.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _git_command(git_dir: Path, workspace: Path) -> list[str]:
    return [
        "git",
        f"--git-dir={git_dir}",
        f"--work-tree={workspace}",
    ]


def _run_git(
    command: list[str],
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )


def _git_output(
    command: list[str],
    *,
    check: bool = True,
) -> str:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
        check=check,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
