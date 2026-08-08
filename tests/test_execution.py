"""Tests for tools/execution.py — the shared tool-calling dispatcher."""

from collections.abc import Mapping

from langchain_core.tools import StructuredTool

from audit_agent.tools.execution import execute_tool


def _echo_tool() -> StructuredTool:
    def echo(msg: str) -> str:
        """Return the input message unchanged."""
        return msg

    return StructuredTool.from_function(func=echo, name="echo")


def _failing_tool() -> StructuredTool:
    def fail() -> str:
        """Always raise a RuntimeError."""
        raise RuntimeError("tool explosion")

    return StructuredTool.from_function(func=fail, name="fail")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_execute_tool_invokes_registered_tool() -> None:
    tool = _echo_tool()
    result = execute_tool(
        {"name": "echo", "args": {"msg": "hello"}},
        tools_by_name={"echo": tool},
    )

    assert result == "hello"


def test_execute_tool_returns_error_when_missing_args_validation_fails() -> None:
    tool = _echo_tool()
    result = execute_tool(
        {"name": "echo"},
        tools_by_name={"echo": tool},
    )

    # Pydantic schema requires "msg"; empty dict fails validation
    assert isinstance(result, Mapping)
    assert "error" in result
    assert "ValidationError" in result["error"]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_execute_tool_returns_error_when_tool_name_unknown() -> None:
    result = execute_tool(
        {"name": "nonexistent", "args": {}},
        tools_by_name={},
    )

    assert isinstance(result, Mapping)
    assert "error" in result
    assert "Unknown tool" in result["error"]


def test_execute_tool_returns_error_when_args_is_not_a_mapping() -> None:
    tool = _echo_tool()
    result = execute_tool(
        {"name": "echo", "args": ["not", "a", "mapping"]},
        tools_by_name={"echo": tool},
    )

    assert isinstance(result, Mapping)
    assert "error" in result
    assert "Invalid arguments" in result["error"]


def test_execute_tool_returns_error_when_tool_raises() -> None:
    tool = _failing_tool()
    result = execute_tool(
        {"name": "fail", "args": {}},
        tools_by_name={"fail": tool},
    )

    assert isinstance(result, Mapping)
    assert "error" in result
    assert "RuntimeError" in result["error"]
    assert "tool explosion" in result["error"]


def test_execute_tool_handles_missing_name_key() -> None:
    tool = _echo_tool()
    result = execute_tool(
        {"args": {"msg": "hi"}},
        tools_by_name={"echo": tool},
    )

    assert isinstance(result, Mapping)
    assert "error" in result
    assert "Unknown tool" in result["error"]
