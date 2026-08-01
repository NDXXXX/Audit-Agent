import json
from pathlib import Path

import pytest

from ddclaw.core.state import RuntimeState
from ddclaw.graph.memory import (
    RULES_LAYER,
    _short_text,
    _trim_handoffs,
    build_layered_memory,
    format_layered_memory_for_prompt,
    memory_event,
    read_history_summary,
    read_notepad,
)


def test_memory_file_readers_do_not_create_missing_files(tmp_path: Path) -> None:
    runtime = RuntimeState.create(tmp_path)

    assert read_notepad(runtime) == {
        "path": "NOTEPAD.md",
        "exists": False,
        "content": "",
    }
    assert read_history_summary(runtime) == {
        "path": "HISTORY_SUMMARY.md",
        "exists": False,
        "content": "",
    }
    assert not (tmp_path / "NOTEPAD.md").exists()
    assert not (tmp_path / "HISTORY_SUMMARY.md").exists()


def test_build_layered_memory_reads_files_and_bounds_context(
    tmp_path: Path,
) -> None:
    runtime = RuntimeState.create(tmp_path)
    (tmp_path / "NOTEPAD.md").write_text("durable note", encoding="utf-8")
    (tmp_path / "HISTORY_SUMMARY.md").write_text(
        "history from disk",
        encoding="utf-8",
    )
    handoffs = [
        {
            "from_agent": "planner",
            "to_agent": "codeAgent",
            "instruction": f"instruction-{index}",
            "result": f"result-{index}",
        }
        for index in range(8)
    ]
    compression_events = [
        {"node": "planner", "session_turn": index}
        for index in range(5)
    ]

    memory = build_layered_memory(
        {
            "runtime": runtime,
            "task": "Create a researched implementation",
            "session_id": "session-123",
            "session_turn": 7,
            "session_context": "prior conversation",
            "plan_summary": "Implement and verify",
            "todos": [{"id": "1", "status": "pending"}],
            "acceptance_criteria": ["Tests pass"],
            "verification_commands": ["python -m pytest"],
            "research_notes": "r" * 1700,
            "sources": [
                {
                    "title": "Official docs",
                    "url": "https://example.com/docs",
                    "content": "must not enter prompt memory",
                    "score": 0.99,
                }
            ],
            "agent_handoffs": handoffs,
            "code_agent_summary": "c" * 1100,
            "verifier_summary": "v" * 1100,
            "last_error": "e" * 1500,
            "attempts": 2,
            "max_attempts": 4,
            "history_summary": "state history",
            "context_summary": "context summary",
            "compression_events": compression_events,
        },
        node="planner",
    )

    assert memory["rules"] == RULES_LAYER
    assert memory["rules"] is not RULES_LAYER
    working = memory["working_memory"]
    assert working["node"] == "planner"
    assert working["session_id"] == "session-123"
    assert working["session_turn"] == 7
    assert working["session_context"] == "prior conversation"
    assert len(working["research_notes"]) == 1603
    assert working["research_notes"].endswith("...")
    assert len(working["code_agent_summary"]) == 1003
    assert len(working["verifier_summary"]) == 1003
    assert len(working["last_error"]) == 1403
    assert working["sources"] == [
        {
            "title": "Official docs",
            "url": "https://example.com/docs",
        }
    ]
    assert [item["instruction"] for item in working["agent_handoffs"]] == [
        f"instruction-{index}" for index in range(2, 8)
    ]

    history = memory["history_summary_store"]
    assert history["history_exists"] is True
    assert history["history_summary"] == "state history"
    assert history["notepad_exists"] is True
    assert history["notepad"] == "durable note"
    assert history["compression_events"] == compression_events[-3:]


def test_history_file_is_used_when_state_summary_is_empty(tmp_path: Path) -> None:
    runtime = RuntimeState.create(tmp_path)
    (tmp_path / "HISTORY_SUMMARY.md").write_text(
        "persisted history",
        encoding="utf-8",
    )

    memory = build_layered_memory({"runtime": runtime})

    assert memory["history_summary_store"]["history_summary"] == (
        "persisted history"
    )


def test_short_text_and_handoff_trimming_helpers() -> None:
    assert _short_text("short", 5) == "short"
    assert _short_text("abcdef", 4) == "abcd..."
    assert _short_text("abcdef", 3) == "abc..."
    assert _short_text("abcdef", 0) == "..."
    with pytest.raises(ValueError, match="limit"):
        _short_text("text", -1)

    handoffs = [{"instruction": str(index)} for index in range(9)]
    trimmed = _trim_handoffs(handoffs)
    assert [item["instruction"] for item in trimmed] == [
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
    ]
    trimmed[0]["instruction"] = "changed"
    assert handoffs[3]["instruction"] == "3"


def test_format_layered_memory_for_prompt_preserves_unicode(
    tmp_path: Path,
) -> None:
    memory = build_layered_memory(
        {
            "runtime": RuntimeState.create(tmp_path),
            "task": "中文任务",
        }
    )

    formatted = format_layered_memory_for_prompt(memory)

    assert "中文任务" in formatted
    assert json.loads(formatted) == memory


def test_memory_event_identifies_node_and_snapshot(tmp_path: Path) -> None:
    memory = build_layered_memory(
        {
            "runtime": RuntimeState.create(tmp_path),
            "task": "Inject memory",
        },
        node="planner",
    )

    assert memory_event(memory, node="planner") == {
        "type": "memory",
        "node": "planner",
        "memory": memory,
    }
