"""Persistent conversation sessions scoped to one Audit Agent workspace."""

from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from audit_agent.core.paths import ensure_workspace

SESSION_ROOT = ".audit/session"
SESSION_FILE = "session.json"
SESSION_SUMMARY_FILE = "SESSION_SUMMARY.md"
MAX_SESSION_CONTEXT = 7000
MAX_TURN_CONTENT = 4000

_RECENT_CONTEXT_TURNS = 10
_RECENT_WORKSPACE_FILES = 30
_CONTEXT_TURN_CONTENT = 120
_CONTEXT_FILE_PATH = 80
_IGNORED_FILE_PARTS = {
    ".git",
    ".audit",
    ".pytest_cache",
    ".venv",
    "__pycache__",
}


def load_or_create_session(workspace: Path) -> dict[str, Any]:
    """Load a workspace session, creating a valid persisted session if needed."""

    root = _session_root(workspace)
    path = root / SESSION_FILE
    payload: Mapping[str, Any] | None = None
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, Mapping):
            payload = loaded

    session = _normalize_session(payload)
    summary_path = root / SESSION_SUMMARY_FILE
    if payload is None or dict(payload) != session or not summary_path.is_file():
        return save_session(workspace, session)
    return session


def append_user_turn(session: MutableMapping[str, Any], content: str) -> int:
    """Append one user input and return its monotonically increasing turn ID."""

    _require_mutable_session(session)
    current_turn = _nonnegative_int(session.get("turn_index"), default=0)
    turn = current_turn + 1
    timestamp = _utc_now()
    turns = _turn_list(session)
    turns.append(
        {
            "turn": turn,
            "role": "user",
            "content": _limit_turn_content(content),
            "timestamp": timestamp,
        }
    )
    session["turn_index"] = turn
    session["updated_at"] = timestamp
    return turn


def append_assistant_turn(
    session: MutableMapping[str, Any],
    *,
    turn: int,
    route: str,
    content: str,
    summary: str = "",
) -> None:
    """Append one assistant response associated with a user turn."""

    _require_mutable_session(session)
    if route not in {"chat", "workflow"}:
        raise ValueError("route must be either 'chat' or 'workflow'")
    if isinstance(turn, bool) or not isinstance(turn, int) or turn < 1:
        raise ValueError("turn must be a positive integer")

    timestamp = _utc_now()
    turns = _turn_list(session)
    turns.append(
        {
            "turn": turn,
            "role": "assistant",
            "route": route,
            "content": _limit_turn_content(content),
            "summary": _limit_turn_content(summary),
            "timestamp": timestamp,
        }
    )
    session["turn_index"] = max(
        _nonnegative_int(session.get("turn_index"), default=0),
        turn,
    )
    session["updated_at"] = timestamp


def save_session(
    workspace: Path,
    session: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist normalized JSON state and its human-readable Markdown summary."""

    normalized = _normalize_session(session)
    normalized["updated_at"] = _utc_now()
    root = _session_root(workspace)
    root.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(
        root / SESSION_FILE,
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
    )
    _write_text_atomic(
        root / SESSION_SUMMARY_FILE,
        _build_session_summary(normalized),
    )

    if isinstance(session, MutableMapping):
        session.clear()
        session.update(normalized)
        return dict(session)
    return normalized


def build_session_context(
    workspace: Path,
    session: Mapping[str, Any] | None = None,
) -> str:
    """Build bounded session and workspace context for lightweight LLM nodes."""

    root = ensure_workspace(workspace)
    normalized = (
        load_or_create_session(root)
        if session is None
        else _normalize_session(session)
    )
    files = _recent_workspace_files(root)
    turns = _recent_logical_turns(normalized.get("recent_turns", []))

    lines = [
        "Session context",
        f"- session_id: {normalized['session_id']}",
        f"- turn_index: {normalized['turn_index']}",
        "",
        f"Workspace files (up to {_RECENT_WORKSPACE_FILES}, most recent first):",
    ]
    if files:
        lines.extend(
            f"- {_truncate(item['path'], _CONTEXT_FILE_PATH)} "
            f"({item['size']} bytes)"
            for item in files
        )
    else:
        lines.append("- No workspace files found.")

    lines.extend(
        [
            "",
            f"Recent conversation (up to {_RECENT_CONTEXT_TURNS} turns):",
        ]
    )
    if turns:
        for item in turns:
            turn = item.get("turn", 0)
            role = str(item.get("role") or "unknown")
            route = str(item.get("route") or "")
            route_suffix = f"/{route}" if route else ""
            content = _truncate(
                _single_line(
                    str(item.get("summary") or item.get("content") or "")
                ),
                _CONTEXT_TURN_CONTENT,
            )
            lines.append(
                f"- Turn {turn} [{role}{route_suffix}]: "
                f"{content}"
            )
    else:
        lines.append("- No prior conversation.")

    return _truncate("\n".join(lines), MAX_SESSION_CONTEXT)


def _normalize_session(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    now = _utc_now()
    source = payload if isinstance(payload, Mapping) else {}
    raw_session_id = source.get("session_id")
    session_id = (
        raw_session_id.strip()
        if isinstance(raw_session_id, str) and raw_session_id.strip()
        else str(uuid4())
    )
    created_at = _timestamp_or_default(source.get("created_at"), now)
    updated_at = _timestamp_or_default(source.get("updated_at"), created_at)
    turns = _normalize_turns(source.get("recent_turns"))
    highest_turn = max(
        (item["turn"] for item in turns),
        default=0,
    )
    turn_index = max(
        _nonnegative_int(source.get("turn_index"), default=0),
        highest_turn,
    )
    return {
        "session_id": session_id,
        "turn_index": turn_index,
        "recent_turns": turns,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _normalize_turns(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    turns: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        turn = _nonnegative_int(raw.get("turn"), default=-1)
        role = raw.get("role")
        if turn < 1 or role not in {"user", "assistant"}:
            continue
        item: dict[str, Any] = {
            "turn": turn,
            "role": role,
            "content": _limit_turn_content(raw.get("content", "")),
            "timestamp": _timestamp_or_default(raw.get("timestamp"), _utc_now()),
        }
        if role == "assistant":
            route = raw.get("route")
            item["route"] = route if route in {"chat", "workflow"} else "workflow"
            item["summary"] = _limit_turn_content(raw.get("summary", ""))
        turns.append(item)
    return turns


def _turn_list(session: MutableMapping[str, Any]) -> list[dict[str, Any]]:
    turns = session.get("recent_turns")
    if not isinstance(turns, list):
        turns = _normalize_turns(turns)
        session["recent_turns"] = turns
    return turns


def _recent_logical_turns(value: Any) -> list[dict[str, Any]]:
    turns = _normalize_turns(value)
    selected_ids: list[int] = []
    for item in reversed(turns):
        turn = item["turn"]
        if turn not in selected_ids:
            selected_ids.append(turn)
        if len(selected_ids) >= _RECENT_CONTEXT_TURNS:
            break
    allowed = set(selected_ids)
    return [item for item in turns if item["turn"] in allowed]


def _recent_workspace_files(workspace: Path) -> list[dict[str, Any]]:
    candidates: list[tuple[int, str, int]] = []
    for path in workspace.rglob("*"):
        try:
            relative = path.relative_to(workspace)
            if any(part in _IGNORED_FILE_PARTS for part in relative.parts):
                continue
            if path.is_symlink() or not path.is_file():
                continue
            stat = path.stat()
        except OSError:
            continue
        candidates.append(
            (stat.st_mtime_ns, relative.as_posix(), stat.st_size)
        )
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [
        {"path": path, "size": size}
        for _, path, size in candidates[:_RECENT_WORKSPACE_FILES]
    ]


def _build_session_summary(session: Mapping[str, Any]) -> str:
    lines = [
        "# Audit Agent Session Summary",
        "",
        f"- Session ID: `{session.get('session_id', '')}`",
        f"- Turn index: {session.get('turn_index', 0)}",
        f"- Created: {session.get('created_at', '')}",
        f"- Updated: {session.get('updated_at', '')}",
        "",
        "## Recent conversation",
        "",
    ]
    turns = _recent_logical_turns(session.get("recent_turns", []))
    if not turns:
        lines.append("No conversation turns have been recorded.")
    for item in turns:
        role = str(item.get("role", "unknown")).capitalize()
        route = item.get("route")
        route_text = f" ({route})" if route else ""
        content = str(item.get("summary") or item.get("content") or "")
        lines.extend(
            [
                f"### Turn {item.get('turn', 0)} — {role}{route_text}",
                "",
                _single_line(content) or "_(empty)_",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _session_root(workspace: Path) -> Path:
    return ensure_workspace(workspace) / SESSION_ROOT


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _require_mutable_session(session: MutableMapping[str, Any]) -> None:
    if not isinstance(session, MutableMapping):
        raise TypeError("session must be a mutable mapping")
    normalized = _normalize_session(session)
    session.clear()
    session.update(normalized)


def _limit_turn_content(value: Any) -> str:
    return _truncate(value if isinstance(value, str) else str(value), MAX_TURN_CONTENT)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 3:
        return "." * limit
    return f"{text[: limit - 3]}..."


def _single_line(text: str) -> str:
    return " ".join(text.split())


def _nonnegative_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _timestamp_or_default(value: Any, default: str) -> str:
    return value if isinstance(value, str) and value.strip() else default


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
