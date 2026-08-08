"""Create LangChain tools bound to one runtime state."""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from audit_agent.core.state import RuntimeState
from audit_agent.tools.bash_tool import BashInput, BashTool
from audit_agent.tools.file_tools import (
    FileEditInput,
    FileEditTool,
    FileReadInput,
    FileReadTool,
    FileWriteInput,
    FileWriteTool,
)
from audit_agent.tools.grep_tool import GrepInput, GrepTool


def build_tools(state: RuntimeState) -> list[StructuredTool]:
    """Return all model-callable tools bound to *state*."""

    file_read = FileReadTool(state)
    file_write = FileWriteTool(state)
    file_edit = FileEditTool(state)
    grep = GrepTool(state)
    bash = BashTool(state)

    return [
        StructuredTool.from_function(
            func=file_read.run,
            name="file_read",
            description=(
                "Read a UTF-8 text file inside the workspace. offset is a "
                "zero-based line number and limit is the maximum line count."
            ),
            args_schema=FileReadInput,
        ),
        StructuredTool.from_function(
            func=file_write.run,
            name="file_write",
            description=(
                "Create or completely overwrite a UTF-8 text file inside the "
                "workspace, creating parent directories as needed."
            ),
            args_schema=FileWriteInput,
        ),
        StructuredTool.from_function(
            func=file_edit.run,
            name="file_edit",
            description=(
                "Replace one unique literal text fragment in a workspace file. "
                "The operation fails when there are zero or multiple matches."
            ),
            args_schema=FileEditInput,
        ),
        StructuredTool.from_function(
            func=grep.run,
            name="grep",
            description=(
                "Search files in the workspace with a Python regular expression "
                "and return matching lines as path:line:content."
            ),
            args_schema=GrepInput,
        ),
        StructuredTool.from_function(
            func=bash.run_bash,
            name="bash",
            description=(
                "Run a shell command with the workspace as its current directory "
                "and enforce a timeout. Risky install, download, and server "
                "commands are subject to the configured approval policy."
            ),
            args_schema=BashInput,
        ),
    ]


def build_read_only_tools(state: RuntimeState) -> list[StructuredTool]:
    """Return file inspection tools that cannot mutate the workspace."""

    file_read = FileReadTool(state)
    grep = GrepTool(state)
    return [
        StructuredTool.from_function(
            func=file_read.run,
            name="file_read",
            description=(
                "Read a UTF-8 text file inside the workspace. offset is a "
                "zero-based line number and limit is the maximum line count."
            ),
            args_schema=FileReadInput,
        ),
        StructuredTool.from_function(
            func=grep.run,
            name="grep",
            description=(
                "Search files in the workspace with a Python regular expression "
                "and return matching lines as path:line:content."
            ),
            args_schema=GrepInput,
        ),
    ]
