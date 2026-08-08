"""Read, write, and edit files inside a configured workspace."""

from __future__ import annotations

from pydantic import BaseModel, Field

from audit_agent.core.state import RuntimeState


class FileReadInput(BaseModel):
    """Arguments accepted by the file-read tool."""

    file_path: str = Field(description="Path to a file, relative to the workspace.")
    offset: int = Field(
        default=0,
        ge=0,
        description="Zero-based line offset at which reading starts.",
    )
    limit: int = Field(
        default=2000,
        ge=1,
        description="Maximum number of lines to return.",
    )


class FileWriteInput(BaseModel):
    """Arguments accepted by the file-write tool."""

    file_path: str = Field(description="Path to a file, relative to the workspace.")
    content: str = Field(description="Complete UTF-8 text to write to the file.")


class FileEditInput(BaseModel):
    """Arguments accepted by the file-edit tool."""

    file_path: str = Field(description="Path to a file, relative to the workspace.")
    old_text: str = Field(description="The unique text fragment to replace.")
    new_text: str = Field(description="Replacement text.")


class FileReadTool:
    """Read a range of lines from a UTF-8 text file."""

    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def __call__(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        return self.run(file_path=file_path, offset=offset, limit=limit)

    def run(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        if offset < 0:
            raise ValueError("offset must be greater than or equal to 0")
        if limit < 1:
            raise ValueError("limit must be greater than or equal to 1")

        target = self.state.resolve_path(file_path, must_exist=True)
        if not target.is_file():
            raise IsADirectoryError(f"Not a file: {file_path}")

        with target.open("r", encoding="utf-8") as stream:
            lines = stream.readlines()
        return "".join(lines[offset : offset + limit])


class FileWriteTool:
    """Create or overwrite a UTF-8 text file."""

    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def __call__(self, file_path: str, content: str) -> str:
        return self.run(file_path=file_path, content=content)

    def run(self, file_path: str, content: str) -> str:
        target = self.state.resolve_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Resolve once more after creating parents, so a concurrently introduced
        # or pre-existing symlink cannot redirect the final write.
        target = self.state.resolve_path(target)
        if target.exists() and target.is_dir():
            raise IsADirectoryError(f"Cannot overwrite a directory: {file_path}")

        target.write_text(content, encoding="utf-8")
        relative = target.relative_to(self.state.workspace).as_posix()
        return f"Wrote {len(content)} characters to {relative}"


class FileEditTool:
    """Replace exactly one occurrence of a text fragment in a file."""

    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def __call__(self, file_path: str, old_text: str, new_text: str) -> str:
        return self.run(
            file_path=file_path,
            old_text=old_text,
            new_text=new_text,
        )

    def run(self, file_path: str, old_text: str, new_text: str) -> str:
        if not old_text:
            raise ValueError("old_text must not be empty")

        target = self.state.resolve_path(file_path, must_exist=True)
        if not target.is_file():
            raise IsADirectoryError(f"Not a file: {file_path}")

        content = target.read_text(encoding="utf-8")
        match_count = content.count(old_text)
        if match_count == 0:
            raise ValueError(f"old_text was not found in {file_path}")
        if match_count > 1:
            raise ValueError(
                f"old_text must be unique in {file_path}; found {match_count} matches"
            )

        target.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        relative = target.relative_to(self.state.workspace).as_posix()
        return f"Replaced one occurrence in {relative}"
