import json
from pathlib import Path

import pytest

from audit_agent.core.session import (
    MAX_SESSION_CONTEXT,
    MAX_TURN_CONTENT,
    SESSION_FILE,
    SESSION_ROOT,
    SESSION_SUMMARY_FILE,
    append_assistant_turn,
    append_user_turn,
    build_session_context,
    load_or_create_session,
    save_session,
)


def test_load_or_create_session_persists_complete_initial_state(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"

    session = load_or_create_session(workspace)

    assert session["session_id"]
    assert session["turn_index"] == 0
    assert session["recent_turns"] == []
    assert session["created_at"]
    assert session["updated_at"]
    session_root = workspace / SESSION_ROOT
    assert json.loads(
        (session_root / SESSION_FILE).read_text(encoding="utf-8")
    ) == session
    summary = (session_root / SESSION_SUMMARY_FILE).read_text(encoding="utf-8")
    assert "# Audit Agent Session Summary" in summary
    assert session["session_id"] in summary

    assert load_or_create_session(workspace) == session


def test_append_turns_and_save_session(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    session = load_or_create_session(workspace)

    turn = append_user_turn(session, "请继续实现项目")
    append_assistant_turn(
        session,
        turn=turn,
        route="workflow",
        content="实现和测试已完成",
        summary="完成 session 模块",
    )
    saved = save_session(workspace, session)

    assert turn == 1
    assert saved["turn_index"] == 1
    assert saved["recent_turns"][0]["role"] == "user"
    assert saved["recent_turns"][1]["route"] == "workflow"
    assert load_or_create_session(workspace) == saved
    summary = (
        workspace / SESSION_ROOT / SESSION_SUMMARY_FILE
    ).read_text(encoding="utf-8")
    assert "请继续实现项目" in summary
    assert "完成 session 模块" in summary


def test_turn_content_is_bounded_and_invalid_assistant_values_fail(
    tmp_path: Path,
) -> None:
    session = load_or_create_session(tmp_path / "workspace")
    turn = append_user_turn(session, "x" * (MAX_TURN_CONTENT + 100))

    assert len(session["recent_turns"][0]["content"]) == MAX_TURN_CONTENT
    assert session["recent_turns"][0]["content"].endswith("...")
    with pytest.raises(ValueError, match="route"):
        append_assistant_turn(
            session,
            turn=turn,
            route="invalid",
            content="no",
        )
    with pytest.raises(ValueError, match="positive integer"):
        append_assistant_turn(
            session,
            turn=0,
            route="chat",
            content="no",
        )


def test_load_repairs_invalid_session_json(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / SESSION_ROOT
    root.mkdir(parents=True)
    (root / SESSION_FILE).write_text("not json", encoding="utf-8")

    session = load_or_create_session(workspace)

    assert session["turn_index"] == 0
    assert session["recent_turns"] == []
    assert json.loads((root / SESSION_FILE).read_text(encoding="utf-8")) == session


def test_legacy_session_storage_is_left_untouched_and_ignored(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    legacy_root = workspace / ("." + "nd" + "claw") / "session"
    legacy_root.mkdir(parents=True)
    legacy_payload = {
        "session_id": "legacy-session",
        "turn_index": 99,
        "recent_turns": [],
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
    }
    legacy_file = legacy_root / SESSION_FILE
    legacy_file.write_text(json.dumps(legacy_payload), encoding="utf-8")

    session = load_or_create_session(workspace)

    assert session["session_id"] != "legacy-session"
    assert session["turn_index"] == 0
    assert legacy_file.read_text(encoding="utf-8") == json.dumps(legacy_payload)
    assert (workspace / SESSION_ROOT / SESSION_FILE).is_file()


def test_session_context_has_recent_files_and_ten_logical_turns(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for index in range(35):
        (workspace / f"file-{index:02}.txt").write_text(
            str(index),
            encoding="utf-8",
        )
    internal = workspace / ".audit" / "ignored.txt"
    internal.parent.mkdir()
    internal.write_text("secret", encoding="utf-8")
    session = load_or_create_session(workspace)
    for index in range(12):
        turn = append_user_turn(session, f"user-{index}")
        append_assistant_turn(
            session,
            turn=turn,
            route="chat",
            content=f"assistant-{index}",
        )

    context = build_session_context(workspace, session)

    assert session["session_id"] in context
    assert "turn_index: 12" in context
    assert "Turn 1 [user]:" not in context
    assert "Turn 2 [user]:" not in context
    assert "Turn 3 [user]: user-2" in context
    assert "assistant-11" in context
    assert context.count(" bytes)") == 30
    assert ".audit" not in context
    assert len(context) <= MAX_SESSION_CONTEXT


def test_session_context_enforces_total_length_limit(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    session = load_or_create_session(workspace)
    for _ in range(10):
        turn = append_user_turn(session, "用户内容" * 1500)
        append_assistant_turn(
            session,
            turn=turn,
            route="workflow",
            content="助手内容" * 1500,
        )

    context = build_session_context(workspace, session)

    assert len(context) <= MAX_SESSION_CONTEXT
    assert "Turn 1 [user]:" in context
    assert "Turn 10 [assistant/workflow]:" in context
