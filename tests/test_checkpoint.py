import json
import subprocess
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from audit_agent.core.checkpoint import (
    CheckpointManager,
    build_recovery_markdown,
    normalize_checkpoint_mode,
    resume_command,
    workspace_manifest,
)
from audit_agent.core.state import RuntimeState


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (None, "light"),
        ("", "light"),
        ("invalid", "light"),
        (" LIGHT ", "light"),
        ("STRICT", "strict"),
        ("off", "off"),
    ],
)
def test_normalize_checkpoint_mode(mode: str | None, expected: str) -> None:
    assert normalize_checkpoint_mode(mode) == expected


def test_light_checkpoint_saves_snapshot_and_restores_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    runtime = RuntimeState.create(workspace, checkpoint_mode="light")
    source = workspace / "app.py"
    source.write_text("print('checkpoint')\n", encoding="utf-8")
    manager = CheckpointManager(runtime, task="Create app")
    state = {
        "task": "Create app",
        "runtime": runtime,
        "messages": [HumanMessage(content="Create app.py")],
        "todos": [
            {
                "id": "todo-1",
                "content": "Create app.py",
                "status": "completed",
                "note": "Done",
            }
        ],
        "context_summary": "app.py was created and still needs verification.",
        "attempts": 1,
        "passed": False,
    }

    event = manager.save(
        state,
        latest_node="planner",
        event={"type": "node_update"},
    )

    assert manager.enabled is True
    assert manager.root == workspace / ".audit" / "checkpoints"
    assert event is not None
    assert event["type"] == "checkpoint_saved"
    assert event["latest_node"] == "planner"

    checkpoint_path = manager.root / "checkpoint.json"
    recovery_path = manager.root / "RECOVERY.md"
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert payload["checkpoint_mode"] == "light"
    assert payload["git_commit"]
    assert payload["workspace_manifest"] == [
        {
            "path": "app.py",
            "type": "file",
            "size": len("print('checkpoint')\n"),
            "sha256": payload["workspace_manifest"][0]["sha256"],
        }
    ]
    assert not (manager.root / "state.json").exists()
    assert not (manager.root / "events.jsonl").exists()
    recovery = recovery_path.read_text(encoding="utf-8")
    assert "# Audit Agent Recovery" in recovery
    assert "Create app" in recovery
    assert "app.py" in recovery
    assert payload["git_commit"] in recovery
    assert resume_command(workspace) in recovery

    source.write_text("print('broken')\n", encoding="utf-8")
    inputs, resume_event = CheckpointManager.load_resume_inputs(
        runtime,
        max_attempts=5,
    )

    assert source.read_text(encoding="utf-8") == "print('checkpoint')\n"
    assert inputs["task"] == "Create app"
    assert inputs["runtime"] is runtime
    assert inputs["attempts"] == 1
    assert inputs["max_attempts"] == 5
    assert len(inputs["messages"]) == 1
    assert isinstance(inputs["messages"][0], AIMessage)
    assert "still needs verification" in inputs["messages"][0].content
    assert resume_event["type"] == "checkpoint_resumed"
    assert resume_event["restored_workspace"] is True


def test_strict_checkpoint_saves_full_state_and_event_log(
    tmp_path: Path,
) -> None:
    runtime = RuntimeState.create(
        tmp_path / "workspace",
        checkpoint_mode="strict",
    )
    (runtime.workspace / "result.txt").write_text("done", encoding="utf-8")
    manager = CheckpointManager(runtime)
    state = {
        "task": "Strict task",
        "runtime": runtime,
        "messages": [
            HumanMessage(content="Do the work"),
            AIMessage(content="Work completed"),
        ],
        "todos": [],
        "attempts": 2,
        "passed": False,
    }

    manager.save(
        state,
        latest_node="planner",
        event={"type": "tool_call", "name": "file_write"},
    )
    manager.save(
        state,
        latest_node="verifier",
        event={"type": "tool_result", "ok": True},
    )

    strict_state = json.loads(
        (manager.root / "state.json").read_text(encoding="utf-8")
    )
    assert strict_state["task"] == "Strict task"
    assert [item["type"] for item in strict_state["messages"]] == [
        "human",
        "ai",
    ]
    event_lines = (manager.root / "events.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(event_lines) == 2
    assert json.loads(event_lines[0])["event"]["type"] == "tool_call"
    assert json.loads(event_lines[1])["event"]["type"] == "tool_result"

    inputs, resume_event = CheckpointManager.load_resume_inputs(runtime)

    assert inputs["task"] == "Strict task"
    assert inputs["attempts"] == 2
    assert [message.content for message in inputs["messages"]] == [
        "Do the work",
        "Work completed",
    ]
    assert resume_event["checkpoint_mode"] == "strict"


def test_off_checkpoint_mode_does_not_create_files(tmp_path: Path) -> None:
    runtime = RuntimeState.create(
        tmp_path / "workspace",
        checkpoint_mode="off",
    )
    manager = CheckpointManager(runtime, task="Disabled")

    result = manager.save({"task": "Disabled"})

    assert manager.enabled is False
    assert result is None
    assert not manager.root.exists()


def test_manifest_excludes_internal_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / ".audit" / "checkpoints").mkdir(parents=True)
    (workspace / ".git").mkdir()
    (workspace / "src").mkdir()
    (workspace / ".audit" / "checkpoint.json").write_text("internal")
    (workspace / ".git" / "HEAD").write_text("internal")
    (workspace / "src" / "main.py").write_text("print('ok')")

    manifest = workspace_manifest(workspace)

    assert [item["path"] for item in manifest] == ["src/main.py"]


def test_recovery_helpers_include_status_files_commit_and_command(
    tmp_path: Path,
) -> None:
    command = resume_command(tmp_path / "workspace with spaces")
    markdown = build_recovery_markdown(
        {
            "task": "Recover task",
            "status": "running",
            "latest_node": "verifier",
            "checkpoint_mode": "light",
            "git_commit": "abc123",
            "workspace_manifest": [{"path": "app.py", "size": 10}],
            "resume_command": command,
        }
    )

    assert command.startswith("audit --resume ")
    assert "workspace with spaces" in command
    assert "Recover task" in markdown
    assert "running" in markdown
    assert "app.py" in markdown
    assert "abc123" in markdown
    assert command in markdown


def test_resume_requires_existing_checkpoint(tmp_path: Path) -> None:
    runtime = RuntimeState.create(tmp_path / "workspace")

    with pytest.raises(FileNotFoundError, match="Checkpoint does not exist"):
        CheckpointManager.load_resume_inputs(runtime)


def test_strict_finalize_is_terminal_atomic_and_idempotent(
    tmp_path: Path,
) -> None:
    runtime = RuntimeState.create(
        tmp_path / "workspace",
        checkpoint_mode="strict",
    )
    manager = CheckpointManager(runtime, task="Interrupt safely")
    manager.begin_run("trace-interrupt")
    state = {
        "task": "Interrupt safely",
        "runtime": runtime,
        "messages": [HumanMessage(content="run")],
        "attempts": 1,
        "plan_summary": "Preserve this state",
    }
    manager.save(state, status="running", latest_node="planner")

    first = manager.finalize(
        state,
        status="interrupted",
        latest_node="planner",
    )
    second = manager.finalize(
        state,
        status="interrupted",
        latest_node="planner",
    )

    checkpoint = json.loads(
        (manager.root / "checkpoint.json").read_text(encoding="utf-8")
    )
    lease = json.loads((manager.root / "run.json").read_text(encoding="utf-8"))
    strict_state = json.loads(
        (manager.root / "state.json").read_text(encoding="utf-8")
    )
    terminal_events = [
        json.loads(line)
        for line in (manager.root / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if json.loads(line).get("status") == "interrupted"
    ]

    assert first is not None and second is not None
    assert checkpoint["status"] == "interrupted"
    assert checkpoint["latest_node"] == "planner"
    assert checkpoint["state_summary"]["plan_summary"] == "Preserve this state"
    assert lease["status"] == "interrupted"
    assert "runtime" not in strict_state
    assert len(terminal_events) == 1


def test_resume_refuses_active_run_lease(tmp_path: Path) -> None:
    runtime = RuntimeState.create(tmp_path / "workspace")
    manager = CheckpointManager(runtime, task="Active task")
    manager.begin_run("active-trace")
    manager.save({"task": "Active task", "runtime": runtime})

    with pytest.raises(RuntimeError, match="still has an active run"):
        CheckpointManager.load_resume_inputs(runtime)


def test_resume_converts_stale_running_checkpoint_to_interrupted(
    tmp_path: Path,
) -> None:
    runtime = RuntimeState.create(tmp_path / "workspace")
    manager = CheckpointManager(runtime, task="Stale task")
    manager.begin_run("stale-trace")
    manager.save({"task": "Stale task", "runtime": runtime})
    lease_path = manager.root / "run.json"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["pid"] = 999_999_999
    lease_path.write_text(json.dumps(lease), encoding="utf-8")

    _, resume_event = CheckpointManager.load_resume_inputs(runtime)
    checkpoint = json.loads(
        (manager.root / "checkpoint.json").read_text(encoding="utf-8")
    )

    assert resume_event["stale_running"] is True
    assert resume_event["status"] == "interrupted"
    assert resume_event["resume_strategy"] == "replan_from_saved_state"
    assert checkpoint["status"] == "interrupted"


def test_manifest_excludes_regenerated_dependency_and_cache_trees(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    included = workspace / "src" / "main.py"
    excluded = [
        workspace / ".venv" / "bin" / "python",
        workspace / "package" / "__pycache__" / "module.pyc",
        workspace / "node_modules" / "pkg" / "index.js",
        workspace / ".pytest_cache" / "README.md",
    ]
    included.parent.mkdir(parents=True)
    included.write_text("print('ok')", encoding="utf-8")
    for path in excluded:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated", encoding="utf-8")

    manifest = workspace_manifest(workspace)

    assert [item["path"] for item in manifest] == ["src/main.py"]


def test_git_snapshot_excludes_regenerated_dependency_trees(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / ".venv" / "bin").mkdir(parents=True)
    (workspace / ".venv" / "bin" / "python").write_text("large generated")
    (workspace / "src").mkdir()
    (workspace / "src" / "main.py").write_text("print('kept')")
    runtime = RuntimeState.create(workspace)
    manager = CheckpointManager(runtime)

    event = manager.save({"task": "snapshot", "runtime": runtime})
    assert event is not None and event["git_commit"]
    completed = subprocess.run(
        [
            "git",
            f"--git-dir={manager.root / 'workspace.git'}",
            "ls-tree",
            "-r",
            "--name-only",
            str(event["git_commit"]),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.splitlines() == ["src/main.py"]
