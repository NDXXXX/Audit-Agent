import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from ddclaw.agents import code_agent as code_agent_module
from ddclaw.core.state import RuntimeState
from ddclaw.tools.file_tools import FileWriteInput


class FakeBoundCodeModel:
    def __init__(self) -> None:
        self.responses = iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "todo_update",
                            "args": {
                                "id": "todo-1",
                                "status": "in_progress",
                                "note": "Starting implementation.",
                            },
                            "id": "call-todo-start",
                            "type": "tool_call",
                        },
                        {
                            "name": "file_write",
                            "args": {
                                "file_path": "hello.py",
                                "content": "print('hello')\n",
                            },
                            "id": "call-file-write",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "todo_update",
                            "args": {
                                "id": "todo-1",
                                "status": "completed",
                                "note": "Implemented and checked.",
                            },
                            "id": "call-todo-complete",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content=(
                        "Created `hello.py` and checked the generated content."
                    )
                ),
            ]
        )
        self.invocations: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.invocations.append(list(messages))
        return next(self.responses)


class FakeCodeModel:
    def __init__(self, bound: FakeBoundCodeModel) -> None:
        self.bound = bound
        self.tools: list[StructuredTool] | None = None

    def bind_tools(self, tools: list[StructuredTool]) -> FakeBoundCodeModel:
        self.tools = tools
        return self.bound


def test_run_code_agent_executes_tools_and_persists_todos(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = RuntimeState.create(tmp_path / "workspace")
    original_todos = [
        {
            "id": "todo-1",
            "content": "Create hello.py",
            "status": "pending",
            "note": "",
        }
    ]
    state = {
        "task": "Create a greeting program",
        "runtime": runtime,
        "plan_summary": "Implement one Python file.",
        "todos": original_todos,
        "acceptance_criteria": ["hello.py exists"],
        "verification_commands": ["python hello.py"],
        "research_notes": "Use a simple stdout example.",
        "sources": [
            {
                "title": "Python tutorial",
                "url": "https://docs.python.org/3/tutorial/",
            }
        ],
        "last_error": "",
    }
    written_files: list[dict[str, str]] = []

    def file_write(file_path: str, content: str) -> str:
        written_files.append({"file_path": file_path, "content": content})
        return f"Wrote {file_path}"

    file_write_tool = StructuredTool.from_function(
        func=file_write,
        name="file_write",
        description="Write a file.",
        args_schema=FileWriteInput,
    )
    received_runtime: list[RuntimeState] = []

    def fake_build_tools(value: RuntimeState) -> list[StructuredTool]:
        received_runtime.append(value)
        return [file_write_tool]

    bound = FakeBoundCodeModel()
    model = FakeCodeModel(bound)
    monkeypatch.setattr(code_agent_module, "build_tools", fake_build_tools)
    monkeypatch.setattr(code_agent_module, "create_model", lambda: model)
    monkeypatch.setattr(
        code_agent_module,
        "build_layered_memory_snapshot",
        lambda value: {"snapshot": "layered memory"},
    )
    written_events: list[dict[str, Any]] = []

    result = code_agent_module.run_code_agent(
        state,
        "Implement the current todo.",
        writer=written_events.append,
    )

    assert result["ok"] is True
    assert result["summary"] == (
        "Created `hello.py` and checked the generated content."
    )
    assert result["todos"] == [
        {
            "id": "todo-1",
            "content": "Create hello.py",
            "status": "completed",
            "note": "Implemented and checked.",
        }
    ]
    assert original_todos[0]["status"] == "pending"
    assert written_files == [
        {
            "file_path": "hello.py",
            "content": "print('hello')\n",
        }
    ]
    assert received_runtime == [runtime]
    assert model.tools is not None
    assert [tool.name for tool in model.tools] == [
        "file_write",
        "todo_update",
    ]
    assert written_events[0] == {
        "type": "memory",
        "node": "codeAgent",
        "memory": {"snapshot": "layered memory"},
    }
    assert result["tool_events"] == written_events[1:]
    assert [event["type"] for event in written_events] == [
        "memory",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
    ]

    first_messages = bound.invocations[0]
    assert isinstance(first_messages[0], SystemMessage)
    assert first_messages[0].content == code_agent_module.CODE_AGENT_PROMPT
    assert isinstance(first_messages[1], HumanMessage)
    request_text, memory_text = first_messages[1].content.split(
        "\n\nLayered memory:\n",
        1,
    )
    request = json.loads(request_text)
    assert request == {
        "task": "Create a greeting program",
        "instruction": "Implement the current todo.",
        "session_context": {
            "plan_summary": "Implement one Python file.",
            "todos": original_todos,
            "acceptance_criteria": ["hello.py exists"],
            "verification_commands": ["python hello.py"],
            "research_notes": "Use a simple stdout example.",
            "sources": [
                {
                    "title": "Python tutorial",
                    "url": "https://docs.python.org/3/tutorial/",
                }
            ],
            "last_error": "",
        },
    }
    assert json.loads(memory_text) == {"snapshot": "layered memory"}

    tool_messages = [
        message
        for messages in bound.invocations[1:]
        for message in messages
        if isinstance(message, ToolMessage)
    ]
    assert any(
        message.tool_call_id == "call-file-write"
        and json.loads(message.content) == "Wrote hello.py"
        for message in tool_messages
    )


def test_build_layered_memory_snapshot_targets_code_agent(tmp_path: Path) -> None:
    snapshot = code_agent_module.build_layered_memory_snapshot(
        {
            "runtime": RuntimeState.create(tmp_path),
            "task": "Implement",
        }
    )

    assert snapshot["working_memory"]["node"] == "codeAgent"
    assert snapshot["working_memory"]["task"] == "Implement"


def test_run_code_agent_requires_runtime() -> None:
    with pytest.raises(TypeError, match="state.runtime"):
        code_agent_module.run_code_agent(
            {"todos": []},
            "Implement",
        )


def test_run_code_agent_validates_max_loops(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_loops"):
        code_agent_module.run_code_agent(
            {"runtime": RuntimeState.create(tmp_path)},
            "Implement",
            max_loops=0,
        )
