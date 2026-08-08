"""Tests for tools/todo_tools.py — TodoWriteTool and TodoUpdateTool."""

import pytest

from audit_agent.tools.todo_tools import TodoUpdateTool, TodoWriteTool


# ---------------------------------------------------------------------------
# TodoWriteTool
# ---------------------------------------------------------------------------


def test_todo_write_validates_and_returns_complete_plan() -> None:
    tool = TodoWriteTool()
    result = tool.run(
        plan_summary="Add login endpoint",
        todos=[
            {"id": "1", "content": "Write handler", "status": "pending", "note": ""},
            {"id": "2", "content": "Add tests", "status": "pending", "note": ""},
        ],
        acceptance_criteria=["POST /login returns 200"],
        verification_commands=["pytest tests/test_login.py"],
    )

    assert result["plan_summary"] == "Add login endpoint"
    assert len(result["todos"]) == 2
    assert result["todos"][0]["id"] == "1"
    assert result["acceptance_criteria"] == ["POST /login returns 200"]
    assert result["verification_commands"] == ["pytest tests/test_login.py"]


def test_todo_write_rejects_empty_plan_summary() -> None:
    tool = TodoWriteTool()
    with pytest.raises(ValueError):
        tool.run(
            plan_summary="",
            todos=[{"id": "1", "content": "task", "status": "pending", "note": ""}],
            acceptance_criteria=[],
            verification_commands=[],
        )


def test_todo_write_rejects_duplicate_todo_ids() -> None:
    tool = TodoWriteTool()
    with pytest.raises(ValueError, match="unique"):
        tool.run(
            plan_summary="Plan",
            todos=[
                {"id": "dup", "content": "First", "status": "pending", "note": ""},
                {"id": "dup", "content": "Second", "status": "pending", "note": ""},
            ],
            acceptance_criteria=[],
            verification_commands=[],
        )


def test_todo_write_rejects_empty_todo_list() -> None:
    tool = TodoWriteTool()
    with pytest.raises(ValueError):
        tool.run(
            plan_summary="Plan",
            todos=[],
            acceptance_criteria=[],
            verification_commands=[],
        )


def test_todo_write_as_structured_tool() -> None:
    tool = TodoWriteTool()
    structured = tool.as_structured_tool()

    assert structured.name == "todo_write"
    result = structured.invoke(
        {
            "plan_summary": "Plan",
            "todos": [{"id": "1", "content": "task", "status": "pending", "note": ""}],
            "acceptance_criteria": [],
            "verification_commands": [],
        }
    )
    assert result["plan_summary"] == "Plan"


# ---------------------------------------------------------------------------
# TodoUpdateTool
# ---------------------------------------------------------------------------


def test_todo_update_changes_status_and_note() -> None:
    todos = [
        {"id": "1", "content": "task A", "status": "pending", "note": ""},
        {"id": "2", "content": "task B", "status": "pending", "note": ""},
    ]
    tool = TodoUpdateTool(todos)

    result = tool.run(id="1", status="in_progress", note="started")

    assert result["id"] == "1"
    assert result["status"] == "in_progress"
    assert result["note"] == "started"
    # The tool operates on a deep copy, so the original list is not mutated
    assert todos[0]["status"] == "pending"


def test_todo_update_rejects_unknown_id() -> None:
    tool = TodoUpdateTool(
        [{"id": "1", "content": "task", "status": "pending", "note": ""}]
    )

    with pytest.raises(ValueError, match="Unknown todo id"):
        tool.run(id="nonexistent", status="completed")


def test_todo_update_preserves_other_items() -> None:
    todos = [
        {"id": "1", "content": "task A", "status": "pending", "note": ""},
        {"id": "2", "content": "task B", "status": "pending", "note": ""},
    ]
    tool = TodoUpdateTool(todos)

    tool.run(id="1", status="completed")

    assert todos[1]["status"] == "pending"
    assert todos[1]["content"] == "task B"


def test_todo_update_as_structured_tool() -> None:
    todos = [{"id": "1", "content": "task", "status": "pending", "note": ""}]
    tool = TodoUpdateTool(todos)
    structured = tool.as_structured_tool()

    assert structured.name == "todo_update"
    result = structured.invoke({"id": "1", "status": "completed"})
    assert result["status"] == "completed"


def test_todo_update_rejects_empty_id() -> None:
    todos = [{"id": "1", "content": "task", "status": "pending", "note": ""}]
    tool = TodoUpdateTool(todos)
    structured = tool.as_structured_tool()

    with pytest.raises(ValueError):
        structured.invoke({"id": "", "status": "completed"})
