from pathlib import Path

import pytest

from audit_agent.core.paths import WorkspacePathError
from audit_agent.core.state import RuntimeState
from audit_agent.tools.file_tools import FileEditTool, FileReadTool, FileWriteTool


def test_runtime_state_creates_and_canonicalizes_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "new" / "workspace"

    state = RuntimeState(workspace=workspace)

    assert state.workspace == workspace.resolve()
    assert state.workspace.is_dir()


def test_file_tools_write_read_and_edit(tmp_path: Path) -> None:
    state = RuntimeState(workspace=tmp_path / "workspace")
    writer = FileWriteTool(state)
    reader = FileReadTool(state)
    editor = FileEditTool(state)

    writer("nested/example.txt", "zero\none\ntwo\nthree\n")
    assert reader("nested/example.txt", offset=1, limit=2) == "one\ntwo\n"

    editor("nested/example.txt", "two", "TWO")
    assert (state.workspace / "nested/example.txt").read_text() == (
        "zero\none\nTWO\nthree\n"
    )


def test_file_edit_requires_unique_match(tmp_path: Path) -> None:
    state = RuntimeState(workspace=tmp_path)
    target = tmp_path / "repeated.txt"
    target.write_text("same same", encoding="utf-8")

    with pytest.raises(ValueError, match="found 2 matches"):
        FileEditTool(state)("repeated.txt", "same", "new")


def test_file_tools_reject_parent_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    state = RuntimeState(workspace=workspace)

    with pytest.raises(WorkspacePathError):
        FileWriteTool(state)("../outside.txt", "no")


def test_file_tools_reject_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    outside.mkdir()
    state = RuntimeState(workspace=workspace)

    link = workspace / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("Symbolic links are unavailable on this platform")

    with pytest.raises(WorkspacePathError):
        FileWriteTool(state)("escape/leak.txt", "no")
