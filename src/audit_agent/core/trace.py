"""Structured execution tracing for Audit Agent workflow runs."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain_core.messages import BaseMessage, messages_to_dict

if TYPE_CHECKING:
    from audit_agent.core.state import RuntimeState

VALID_TRACE_MODES = {"full", "summary", "off"}
_DEFAULT_TRACE_MODE = "summary"
_TRACE_DIRECTORY = ".audit/traces"
_TRACE_ID_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def normalize_trace_mode(mode: str | None) -> str:
    """Normalize trace mode, defaulting invalid values to ``summary``."""

    normalized = mode.strip().lower() if isinstance(mode, str) else ""
    if normalized == "on":
        return "summary"
    return normalized if normalized in VALID_TRACE_MODES else _DEFAULT_TRACE_MODE


class TraceRecorder:
    """Record graph, tool, approval, checkpoint, and handoff activity."""

    def __init__(self, runtime: RuntimeState, task: str = "") -> None:
        self.runtime = runtime
        self.workspace = runtime.workspace
        self.mode = normalize_trace_mode(runtime.trace_mode)
        self.trace_id = _normalize_trace_id(runtime.trace_id)
        self.root = self.workspace / _TRACE_DIRECTORY / self.trace_id
        self.task = task
        self.node_visits: dict[str, int] = {}
        self.tool_calls = 0
        self.failed_tool_calls = 0
        self.approval_count = 0
        self.checkpoint_count = 0
        self.handoff_count = 0
        self._timeline: list[dict[str, Any]] = []
        self._started_at: str | None = None
        self._started_monotonic: float | None = None
        self._ended_payload: dict[str, Any] | None = None
        self._approval_ids_seen: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def start(
        self,
        inputs: Mapping[str, Any],
        *,
        resumed: bool = False,
        resume_event: Any = None,
    ) -> dict[str, Any] | None:
        """Record the beginning of one workflow run."""

        if not self.enabled:
            return None
        if self._started_at is not None:
            return self._timeline[0]

        self.root.mkdir(parents=True, exist_ok=True)
        self._started_at = _utc_now()
        self._started_monotonic = time.monotonic()
        if not self.task:
            self.task = str(inputs.get("task") or "")
        event = {
            "type": "run_start",
            "trace_id": self.trace_id,
            "task": self.task,
            "resumed": resumed,
            "resume_event": _jsonable(resume_event),
            "inputs": _trace_state(inputs, full=self.mode == "full"),
        }
        return self._record(event)

    def record_custom_event(
        self,
        event: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Record a custom event and update aggregate counters."""

        if not self.enabled:
            return None
        if self._ended_payload is not None:
            return dict(self._ended_payload)
        self._ensure_started()
        payload = _custom_payload(event)
        event_type = str(payload.get("type") or "custom")

        if event_type == "tool_call":
            self.tool_calls += 1
        elif event_type == "tool_result":
            result = payload.get("result")
            result_mapping = result if isinstance(result, Mapping) else {}
            ok = payload.get("ok", result_mapping.get("ok"))
            requires_approval = payload.get(
                "requires_approval",
                result_mapping.get("requires_approval", False),
            )
            if ok is False:
                self.failed_tool_calls += 1
            approval_id = str(
                payload.get("approval_id")
                or result_mapping.get("approval_id")
                or ""
            )
            if (
                bool(requires_approval)
                and (not approval_id or approval_id not in self._approval_ids_seen)
            ):
                self.approval_count += 1
                if approval_id:
                    self._approval_ids_seen.add(approval_id)
        elif event_type == "approval_decision":
            approval_id = str(payload.get("approval_id") or "")
            reused = bool(payload.get("reused"))
            if not reused and (
                not approval_id or approval_id not in self._approval_ids_seen
            ):
                self.approval_count += 1
                if approval_id:
                    self._approval_ids_seen.add(approval_id)
        elif event_type == "handoff":
            self.handoff_count += 1
        elif event_type == "checkpoint_saved":
            self.checkpoint_count += 1

        record = {
            **dict(payload),
            "type": event_type,
            "category": "custom",
        }
        outer_node = event.get("node")
        if outer_node and "stream_node" not in record:
            record["stream_node"] = str(outer_node)
        return self._record(record)

    def record_graph_update(
        self,
        event: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Record one completed graph node update and its visit count."""

        if not self.enabled:
            return None
        if self._ended_payload is not None:
            return dict(self._ended_payload)
        self._ensure_started()
        node = str(event.get("node") or "unknown")
        self.node_visits[node] = self.node_visits.get(node, 0) + 1
        data = event.get("data", event.get("update", {}))
        return self._record(
            {
                "type": "graph_update",
                "category": "graph",
                "node": node,
                "visit": self.node_visits[node],
                "data": _trace_state(
                    data if isinstance(data, Mapping) else {"value": data},
                    full=self.mode == "full",
                ),
            }
        )

    def end(
        self,
        *,
        status: str,
        latest_node: str | None,
        final_state: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Finish tracing and write JSON statistics plus a Markdown timeline."""

        if not self.enabled:
            return None
        if self._ended_payload is not None:
            return dict(self._ended_payload)
        self._ensure_started()
        ended_at = _utc_now()
        duration_ms = max(
            0,
            int((time.monotonic() - (self._started_monotonic or 0)) * 1000),
        )
        self._record(
            {
                "type": "run_end",
                "status": status,
                "latest_node": latest_node,
                "final_state": _trace_state(
                    final_state,
                    full=self.mode == "full",
                ),
            },
            recorded_at=ended_at,
        )

        timeline_head, timeline_tail, timeline_omitted = _timeline_window(
            self._timeline
        )
        payload = {
            "trace_id": self.trace_id,
            "task": self.task,
            "status": status,
            "started_at": self._started_at,
            "ended_at": ended_at,
            "duration_ms": duration_ms,
            "latest_node": latest_node,
            "trace_mode": self.mode,
            "node_visits": dict(self.node_visits),
            "tool_calls": self.tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "approval_count": self.approval_count,
            "checkpoint_count": self.checkpoint_count,
            "handoff_count": self.handoff_count,
            "timeline_head": timeline_head,
            "timeline_tail": timeline_tail,
            "timeline_omitted": timeline_omitted,
        }
        _write_json(self.root / "trace.json", payload)
        (self.root / "timeline.md").write_text(
            build_timeline_markdown(payload, self._timeline),
            encoding="utf-8",
        )
        self._ended_payload = dict(payload)
        return dict(payload)

    def _ensure_started(self) -> None:
        if self._started_at is None:
            self.start({"task": self.task})

    def _record(
        self,
        event: Mapping[str, Any],
        *,
        recorded_at: str | None = None,
    ) -> dict[str, Any]:
        record = {
            "sequence": len(self._timeline) + 1,
            "recorded_at": recorded_at or _utc_now(),
            **_jsonable(dict(event)),
        }
        self._timeline.append(record)
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str))
            stream.write("\n")
        return record


def build_timeline_markdown(
    payload: Mapping[str, Any],
    timeline: Sequence[Mapping[str, Any]],
) -> str:
    """Render a compact human-readable trace timeline."""

    lines = [
        "# Audit Agent Execution Trace",
        "",
        f"- Trace ID: `{payload.get('trace_id', '')}`",
        f"- Task: {payload.get('task', '')}",
        f"- Status: {payload.get('status', '')}",
        f"- Duration: {payload.get('duration_ms', 0)} ms",
        f"- Latest node: {payload.get('latest_node') or 'unknown'}",
        f"- Tool calls: {payload.get('tool_calls', 0)}",
        f"- Failed tool calls: {payload.get('failed_tool_calls', 0)}",
        f"- Approvals: {payload.get('approval_count', 0)}",
        f"- Checkpoints: {payload.get('checkpoint_count', 0)}",
        f"- Handoffs: {payload.get('handoff_count', 0)}",
        "",
        "## Timeline",
        "",
    ]
    for event in timeline:
        sequence = event.get("sequence", "?")
        recorded_at = event.get("recorded_at", "")
        event_type = str(event.get("type") or "event")
        lines.append(
            f"{sequence}. `{recorded_at}` — **{event_type}**"
            f"{_timeline_event_detail(event)}"
        )
    if not timeline:
        lines.append("No events were recorded.")
    lines.append("")
    return "\n".join(lines)


def _custom_payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    if event.get("type") == "custom" and isinstance(event.get("data"), Mapping):
        return event["data"]
    return event


def _timeline_window(
    timeline: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    head = [dict(event) for event in timeline[:20]]
    if len(timeline) <= 100:
        tail = [dict(event) for event in timeline[20:]]
    else:
        tail = [dict(event) for event in timeline[-80:]]
    omitted = max(0, len(timeline) - len(head) - len(tail))
    return head, tail, omitted


def _trace_state(state: Mapping[str, Any], *, full: bool) -> dict[str, Any]:
    if full:
        return {
            str(key): _jsonable(value)
            for key, value in state.items()
            if key != "runtime"
        }
    summary_keys = {
        "task",
        "session_id",
        "session_turn",
        "intent_route",
        "plan_summary",
        "todos",
        "acceptance_criteria",
        "verification_commands",
        "verification_reason",
        "passed",
        "attempts",
        "max_attempts",
        "context_token_count",
        "context_should_compress",
        "context_next_node",
        "final_answer",
    }
    return {
        str(key): _jsonable(value)
        for key, value in state.items()
        if key in summary_keys
    }


def _timeline_event_detail(event: Mapping[str, Any]) -> str:
    event_type = event.get("type")
    if event_type == "graph_update":
        return f" · node `{event.get('node', 'unknown')}`"
    if event_type in {"tool_call", "tool_result"}:
        agent = event.get("agent") or "agent"
        name = event.get("name") or "tool"
        return f" · {agent} / `{name}`"
    if event_type == "handoff":
        return f" · {event.get('from', 'planner')} → {event.get('to', 'agent')}"
    if event_type == "checkpoint_saved":
        return f" · node `{event.get('latest_node') or 'unknown'}`"
    if event_type == "approval_requested":
        return (
            f" · attempt `{event.get('attempt', '?')}` / "
            f"`{event.get('tool_name', 'tool')}`"
        )
    if event_type == "approval_decision":
        return (
            f" · {'approved' if event.get('approved') else 'denied'}"
            f" (reused={bool(event.get('reused'))})"
        )
    if event_type == "run_end":
        return f" · status `{event.get('status', '')}`"
    return ""


def _normalize_trace_id(trace_id: str | None) -> str:
    if isinstance(trace_id, str) and trace_id.strip():
        normalized = _TRACE_ID_PATTERN.sub("-", trace_id.strip()).strip(".-")
        if normalized:
            return normalized[:100]
    return f"trace-{uuid4().hex[:12]}"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseMessage):
        return messages_to_dict([value])[0]
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item) for item in value]
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
