from pathlib import Path

import pytest

from audit_agent.core.state import RuntimeState
from audit_agent.tools.bash_tool import BashTool
from audit_agent.tools.grep_tool import GrepTool
from audit_agent.tools.registry import build_read_only_tools, build_tools


def test_grep_supports_regex_glob_limit_and_case(tmp_path: Path) -> None:
    state = RuntimeState(workspace=tmp_path)
    (tmp_path / "a.py").write_text("Alpha\nbeta\nALPHA\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("alpha\n", encoding="utf-8")

    result = GrepTool(state)(
        pattern=r"^alpha$",
        path=".",
        glob="*.py",
        head_limit=2,
        ignore_case=True,
    )

    assert result.splitlines() == ["a.py:1:Alpha", "a.py:3:ALPHA"]


def test_bash_runs_with_workspace_as_cwd(tmp_path: Path) -> None:
    state = RuntimeState(workspace=tmp_path)

    result = BashTool(state)("pwd")

    assert result == {
        "ok": True,
        "command": "pwd",
        "exit_code": 0,
        "stdout": f"{tmp_path.resolve()}\n",
        "stderr": "",
        "output": str(tmp_path.resolve()),
        "requires_approval": False,
    }


def test_bash_enforces_timeout(tmp_path: Path) -> None:
    state = RuntimeState(workspace=tmp_path)

    with pytest.raises(TimeoutError, match="timed out"):
        BashTool(state)("sleep 2", timeout_seconds=1)


def test_registry_returns_structured_tools(tmp_path: Path) -> None:
    state = RuntimeState(workspace=tmp_path)

    tools = build_tools(state)

    assert [tool.name for tool in tools] == [
        "file_read",
        "file_write",
        "file_edit",
        "grep",
        "bash",
    ]
    tools[1].invoke({"file_path": "from-registry.txt", "content": "ok"})
    assert (tmp_path / "from-registry.txt").read_text(encoding="utf-8") == "ok"


def test_read_only_registry_excludes_mutating_tools(tmp_path: Path) -> None:
    state = RuntimeState(workspace=tmp_path)

    tools = build_read_only_tools(state)

    assert [tool.name for tool in tools] == ["file_read", "grep"]
