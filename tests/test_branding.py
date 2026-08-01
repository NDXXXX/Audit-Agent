from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import ddclaw


ROOT = Path(__file__).resolve().parents[1]


def test_lowercase_package_imports() -> None:
    assert ddclaw.__name__ == "ddclaw"


def test_project_exposes_only_new_command_names() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'ddclaw = "ddclaw.cli.app:app"' in pyproject
    assert 'ddclaw-tui = "ddclaw.cli.tui.app:run_tui"' in pyproject

    legacy_command = "ND" + "claw ="
    legacy_tui_command = "ND" + "claw-tui ="
    assert legacy_command not in pyproject
    assert legacy_tui_command not in pyproject


def test_python_module_help_uses_new_package() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-m", "ddclaw", "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout


def test_tracked_project_text_has_no_legacy_brand() -> None:
    forbidden = "nd" + "claw"
    files = [
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / ".gitignore",
        ROOT / ".env.example",
    ]
    files.extend((ROOT / "src" / "ddclaw").rglob("*.py"))
    files.extend((ROOT / "tests").rglob("*.py"))

    occurrences: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if forbidden in text.lower():
            occurrences.append(str(path.relative_to(ROOT)))

    assert occurrences == []
