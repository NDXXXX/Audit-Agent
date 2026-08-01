"""Structured tools for creating and updating graph todo items."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ddclaw.graph.state import TodoItem

TodoStatus = Literal["pending", "in_progress", "completed", "blocked"]


class TodoItemInput(BaseModel):
    """Validated todo item produced by the planner."""

    id: str = Field(min_length=1, description="Stable unique todo identifier.")
    content: str = Field(min_length=1, description="Concrete unit of work.")
    status: TodoStatus = Field(
        default="pending",
        description="Current todo status.",
    )
    note: str = Field(default="", description="Optional status note.")


class TodoWriteInput(BaseModel):
    """Complete structured planning output."""

    plan_summary: str = Field(min_length=1)
    todos: list[TodoItemInput] = Field(min_length=1)
    acceptance_criteria: list[str] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)


class TodoUpdateInput(BaseModel):
    """Update applied to one existing todo item."""

    id: str = Field(min_length=1, description="Todo identifier to update.")
    status: TodoStatus
    note: str = Field(default="", description="Reason or progress note.")


class TodoWriteTool:
    """Validate and return a complete planner payload."""

    name = "todo_write"

    def run(
        self,
        plan_summary: str,
        todos: list[dict[str, Any]],
        acceptance_criteria: list[str],
        verification_commands: list[str],
    ) -> dict[str, Any]:
        payload = TodoWriteInput.model_validate(
            {
                "plan_summary": plan_summary,
                "todos": todos,
                "acceptance_criteria": acceptance_criteria,
                "verification_commands": verification_commands,
            }
        )
        todo_ids = [item.id for item in payload.todos]
        if len(todo_ids) != len(set(todo_ids)):
            raise ValueError("Todo ids must be unique")
        return payload.model_dump()

    def as_structured_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description=(
                "Submit the complete implementation plan, todo list, acceptance "
                "criteria, and verification commands as structured data."
            ),
            args_schema=TodoWriteInput,
        )


class TodoUpdateTool:
    """Update todo statuses while an actor works through a plan."""

    name = "todo_update"

    def __init__(self, todos: list[TodoItem]) -> None:
        self.todos: list[TodoItem] = deepcopy(todos)

    def run(self, id: str, status: TodoStatus, note: str = "") -> dict[str, Any]:
        for item in self.todos:
            if item["id"] == id:
                item["status"] = status
                item["note"] = note
                return dict(item)
        raise ValueError(f"Unknown todo id: {id}")

    def as_structured_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description=(
                "Update the status and note of one existing todo item as work "
                "starts, completes, or becomes blocked."
            ),
            args_schema=TodoUpdateInput,
        )
