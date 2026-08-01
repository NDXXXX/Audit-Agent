"""Regular-expression search within a configured workspace."""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, Field

from ddclaw.core.paths import WorkspacePathError, workspace_relative_path
from ddclaw.core.state import RuntimeState


class GrepInput(BaseModel):
    """Arguments accepted by the grep tool."""

    pattern: str = Field(description="Python regular expression to search for.")
    path: str = Field(
        default=".",
        description="File or directory to search, relative to the workspace.",
    )
    glob: str | None = Field(
        default=None,
        description="Optional file glob, such as '*.py' or 'src/**/*.py'.",
    )
    head_limit: int = Field(
        default=100,
        ge=1,
        description="Maximum number of matching lines to return.",
    )
    ignore_case: bool = Field(
        default=False,
        description="Perform case-insensitive matching.",
    )


class GrepTool:
    """Search UTF-8 text files without leaving the workspace."""

    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def __call__(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        head_limit: int = 100,
        ignore_case: bool = False,
    ) -> str:
        return self.run(
            pattern=pattern,
            path=path,
            glob=glob,
            head_limit=head_limit,
            ignore_case=ignore_case,
        )

    def run(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        head_limit: int = 100,
        ignore_case: bool = False,
    ) -> str:
        if head_limit < 1:
            raise ValueError("head_limit must be greater than or equal to 1")

        try:
            expression = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            raise ValueError(f"Invalid regular expression: {exc}") from exc

        search_root = self.state.resolve_path(path, must_exist=True)
        if not search_root.is_file() and not search_root.is_dir():
            raise ValueError(f"Search path is not a file or directory: {path}")

        matches: list[str] = []
        for candidate in self._iter_files(search_root):
            if glob and not self._matches_glob(candidate, search_root, glob):
                continue
            try:
                safe_candidate = self.state.resolve_path(candidate, must_exist=True)
            except (FileNotFoundError, WorkspacePathError):
                # Skip broken links and symbolic links that leave the workspace.
                continue

            try:
                with safe_candidate.open(
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as stream:
                    for line_number, line in enumerate(stream, start=1):
                        if expression.search(line):
                            relative = workspace_relative_path(
                                self.state.workspace,
                                safe_candidate,
                            )
                            text = line.rstrip("\r\n")
                            matches.append(f"{relative}:{line_number}:{text}")
                            if len(matches) >= head_limit:
                                return "\n".join(matches)
            except (OSError, UnicodeError):
                continue

        return "\n".join(matches) if matches else "No matches found."

    def _iter_files(self, search_root: Path) -> Iterator[Path]:
        if search_root.is_file():
            yield search_root
            return

        for directory, dirnames, filenames in os.walk(
            search_root,
            followlinks=False,
        ):
            dirnames[:] = sorted(
                name
                for name in dirnames
                if not (Path(directory) / name).is_symlink()
            )
            for filename in sorted(filenames):
                candidate = Path(directory) / filename
                if candidate.is_file():
                    yield candidate

    @staticmethod
    def _matches_glob(candidate: Path, search_root: Path, pattern: str) -> bool:
        if search_root.is_file():
            relative = candidate.name
        else:
            relative = candidate.relative_to(search_root).as_posix()
        return fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(
            candidate.name,
            pattern,
        )
