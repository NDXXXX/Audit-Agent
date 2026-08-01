"""Checkpointed and traced event stream for the DDclaw workflow."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from langgraph.graph.message import add_messages

from ddclaw.core.approval import ApprovalHandler
from ddclaw.core.checkpoint import CheckpointManager
from ddclaw.core.session import (
    append_assistant_turn,
    append_user_turn,
    build_session_context,
    load_or_create_session,
    save_session,
)
from ddclaw.core.state import create_runtime
from ddclaw.core.trace import TraceRecorder
from ddclaw.graph.workflow import (
    build_complex_workflow,
    build_entry_workflow,
    build_workflow,
)

_STREAM_MODES = ["updates", "custom"]


def stream_agent_events(
    task: str,
    *,
    workspace: str | PathLike[str],
    max_attempts: int = 3,
    approval_mode: str = "inline",
    approval_handler: ApprovalHandler | None = None,
    checkpoint_mode: str = "light",
    resume_workspace: str | PathLike[str] | None = None,
    trace_mode: str = "on",
) -> Iterator[dict[str, Any]]:
    """Run the workflow with checkpoints and execution tracing enabled."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be greater than or equal to 1")

    runtime = create_runtime(
        workspace,
        approval_mode=approval_mode,
        approval_handler=approval_handler,
        checkpoint_mode=checkpoint_mode,
        resume_from=resume_workspace,
        trace_mode=trace_mode,
    )
    manager = CheckpointManager(runtime, task=task)
    trace = TraceRecorder(runtime, task=task)

    resumed = resume_workspace is not None
    resume_event: dict[str, Any] | None = None
    if resumed:
        inputs, resume_event = manager.load_resume_inputs(
            runtime,
            task=task,
            max_attempts=max_attempts,
        )
    else:
        inputs = _initial_inputs(
            task,
            runtime=runtime,
            max_attempts=max_attempts,
        )

    manager.begin_run(trace.trace_id)
    current_state: dict[str, Any] = dict(inputs)
    latest_node = str(
        (resume_event or {}).get("latest_node") or "start"
    )
    progress = {"latest_node": latest_node}
    try:
        trace.start(
            inputs,
            resumed=resumed,
            resume_event=resume_event,
        )
        _save_and_trace_checkpoint(
            manager,
            trace,
            current_state,
            status="started",
            latest_node=latest_node,
        )
        workflow = build_workflow()
        yield from _stream_workflow_events(
            workflow,
            inputs,
            current_state=current_state,
            manager=manager,
            trace=trace,
            progress=progress,
        )
        latest_node = progress["latest_node"]

        _finish_run(
            manager,
            trace,
            current_state,
            status="finished",
            latest_node=latest_node,
            full_snapshot=True,
        )
    except (KeyboardInterrupt, GeneratorExit):
        latest_node = progress["latest_node"]
        _finish_run(
            manager,
            trace,
            current_state,
            status="interrupted",
            latest_node=latest_node,
            full_snapshot=False,
        )
        raise
    except Exception:
        latest_node = progress["latest_node"]
        _finish_run(
            manager,
            trace,
            current_state,
            status="failed",
            latest_node=latest_node,
            full_snapshot=False,
        )
        raise


def stream_session_events(
    task: str,
    *,
    session_workspace: str | PathLike[str] | None = None,
    max_attempts: int = 3,
    approval_mode: str = "inline",
    approval_handler: ApprovalHandler | None = None,
    checkpoint_mode: str = "light",
    resume_workspace: str | PathLike[str] | None = None,
    trace_mode: str = "on",
) -> Iterator[dict[str, Any]]:
    """Route and run one persistent multi-turn conversation turn."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be greater than or equal to 1")

    workspace = (
        resume_workspace
        or session_workspace
        or Path.cwd() / ".ddclaw-workspace"
    )
    runtime = create_runtime(
        workspace,
        approval_mode=approval_mode,
        approval_handler=approval_handler,
        checkpoint_mode=checkpoint_mode,
        resume_from=resume_workspace,
        trace_mode=trace_mode,
    )
    session = load_or_create_session(runtime.workspace)
    turn = append_user_turn(session, task)
    save_session(runtime.workspace, session)
    session_context = build_session_context(runtime.workspace, session)

    manager = CheckpointManager(runtime, task=task)
    trace = TraceRecorder(runtime, task=task)
    resumed = resume_workspace is not None
    resume_event: dict[str, Any] | None = None
    if resumed:
        inputs, resume_event = manager.load_resume_inputs(
            runtime,
            task=task,
            max_attempts=max_attempts,
        )
    else:
        inputs = _initial_inputs(
            task,
            runtime=runtime,
            max_attempts=max_attempts,
        )
    inputs.update(
        {
            "task": task,
            "runtime": runtime,
            "session_id": session["session_id"],
            "session_turn": turn,
            "session_context": session_context,
        }
    )

    manager.begin_run(trace.trace_id)
    current_state: dict[str, Any] = dict(inputs)
    latest_node = str((resume_event or {}).get("latest_node") or "start")
    progress = {"latest_node": latest_node}
    try:
        trace.start(inputs, resumed=resumed, resume_event=resume_event)
        _save_and_trace_checkpoint(
            manager,
            trace,
            current_state,
            status="started",
            latest_node=latest_node,
        )
        yield from _stream_workflow_events(
            build_entry_workflow(),
            current_state,
            current_state=current_state,
            manager=manager,
            trace=trace,
            progress=progress,
        )

        route = (
            "chat"
            if current_state.get("intent_route") == "chat"
            else "workflow"
        )
        if route == "workflow":
            yield from _stream_workflow_events(
                build_complex_workflow(),
                current_state,
                current_state=current_state,
                manager=manager,
                trace=trace,
                progress=progress,
            )

        final_answer = _session_final_answer(current_state, route=route)
        current_state["final_answer"] = final_answer
        append_assistant_turn(
            session,
            turn=turn,
            route=route,
            content=final_answer,
            summary=_session_assistant_summary(current_state, route=route),
        )
        save_session(runtime.workspace, session)
        latest_node = progress["latest_node"]
        _finish_run(
            manager,
            trace,
            current_state,
            status="finished",
            latest_node=latest_node,
            full_snapshot=True,
        )
    except (KeyboardInterrupt, GeneratorExit):
        latest_node = progress["latest_node"]
        _finish_run(
            manager,
            trace,
            current_state,
            status="interrupted",
            latest_node=latest_node,
            full_snapshot=False,
        )
        raise
    except Exception:
        latest_node = progress["latest_node"]
        _finish_run(
            manager,
            trace,
            current_state,
            status="failed",
            latest_node=latest_node,
            full_snapshot=False,
        )
        raise


def _initial_inputs(
    task: str,
    *,
    runtime: Any,
    max_attempts: int,
) -> dict[str, Any]:
    return {
        "task": task,
        "runtime": runtime,
        "messages": [],
        "todos": [],
        "research_notes": "",
        "sources": [],
        "agent_handoffs": [],
        "attempts": 0,
        "max_attempts": max_attempts,
        "passed": False,
    }


def _stream_workflow_events(
    workflow: Any,
    inputs: Mapping[str, Any],
    *,
    current_state: dict[str, Any],
    manager: CheckpointManager,
    trace: TraceRecorder,
    progress: dict[str, str],
) -> Iterator[dict[str, Any]]:
    """Stream one graph stage into shared state, tracing, and checkpoints."""

    raw_events = workflow.stream(
        dict(inputs),
        stream_mode=_STREAM_MODES,
    )
    for raw_event in raw_events:
        yield from _stream_pending_approval_events(
            current_state,
            manager=manager,
            trace=trace,
            latest_node=progress["latest_node"],
        )
        split_event = _split_stream_event(raw_event)
        if split_event is None:
            continue
        mode, event = split_event

        if mode == "custom":
            if not isinstance(event, Mapping):
                continue
            trace.record_custom_event(event)
            if (
                manager.mode == "strict"
                or _custom_event_needs_checkpoint(event)
            ):
                _save_and_trace_checkpoint(
                    manager,
                    trace,
                    current_state,
                    status="running",
                    latest_node=progress["latest_node"],
                    event=event,
                    snapshot=manager.mode != "strict",
                )
            yield {"type": "custom_event", "event": event}
            continue

        if mode != "updates" or not isinstance(event, Mapping):
            continue
        visited_nodes = _merge_graph_updates(current_state, event)
        if visited_nodes:
            progress["latest_node"] = visited_nodes[-1]
        for node in visited_nodes:
            trace.record_graph_update(
                {
                    "type": "node_update",
                    "node": node,
                    "data": event.get(node),
                }
            )
        _save_and_trace_checkpoint(
            manager,
            trace,
            current_state,
            status="running",
            latest_node=progress["latest_node"],
            event=event,
        )
        yield {"type": "graph_event", "event": event}
    yield from _stream_pending_approval_events(
        current_state,
        manager=manager,
        trace=trace,
        latest_node=progress["latest_node"],
    )


def _session_final_answer(
    state: Mapping[str, Any],
    *,
    route: str,
) -> str:
    content = state.get("final_answer")
    if not content and route == "chat":
        content = state.get("chat_response")
    if not content:
        content = (
            "Workflow finished without a final answer."
            if route == "workflow"
            else "How can I help?"
        )
    return str(content)


def _session_assistant_summary(
    state: Mapping[str, Any],
    *,
    route: str,
) -> str:
    if route == "chat":
        return str(state.get("chat_response") or state.get("final_answer") or "")
    return str(
        state.get("code_agent_summary")
        or state.get("verification_reason")
        or state.get("final_answer")
        or ""
    )


def _split_stream_event(raw_event: Any) -> tuple[str, Any] | None:
    """Accept LangGraph v1 tuples and v2 dictionary stream events."""

    if isinstance(raw_event, tuple) and len(raw_event) == 2:
        return str(raw_event[0]), raw_event[1]
    if isinstance(raw_event, Mapping):
        possible_mode = raw_event.get("type")
        if possible_mode in _STREAM_MODES and "data" in raw_event:
            return str(possible_mode), raw_event["data"]
    return None


def _normalize_graph_event(raw_event: Any) -> Iterator[dict[str, Any]]:
    """Normalize a raw event into the public checkpoint-aware event shape."""

    split_event = _split_stream_event(raw_event)
    if split_event is None:
        return
    mode, event = split_event
    if mode == "custom":
        yield {"type": "custom_event", "event": event}
    elif mode == "updates" and isinstance(event, Mapping):
        yield {"type": "graph_event", "event": event}


def _merge_graph_updates(
    current_state: dict[str, Any],
    event: Mapping[str, Any],
) -> list[str]:
    """Apply streamed node updates using the graph's message reducer."""

    visited_nodes: list[str] = []
    for raw_node, raw_update in event.items():
        node = str(raw_node)
        if not isinstance(raw_update, Mapping):
            continue
        update = dict(raw_update)
        if "messages" in update:
            current_state["messages"] = add_messages(
                current_state.get("messages", []),
                update.pop("messages"),
            )
        current_state.update(update)
        visited_nodes.append(node)
    return visited_nodes


def _custom_event_needs_checkpoint(event: Mapping[str, Any]) -> bool:
    """Return whether a material custom event merits an immediate snapshot."""

    payload = (
        event.get("data")
        if event.get("type") == "custom"
        and isinstance(event.get("data"), Mapping)
        else event
    )
    event_type = payload.get("type") if isinstance(payload, Mapping) else None
    return event_type in {
        "tool_result",
        "search_results",
        "handoff",
        "approval_decision",
        "checkpoint_requested",
    }


def _save_and_trace_checkpoint(
    manager: CheckpointManager,
    trace: TraceRecorder,
    state: Mapping[str, Any],
    *,
    status: str,
    latest_node: str | None,
    event: Any = None,
    snapshot: bool = True,
) -> None:
    checkpoint_event = manager.save(
        state,
        status=status,
        latest_node=latest_node,
        event=event,
        snapshot=snapshot,
    )
    if checkpoint_event is not None:
        trace.record_custom_event(checkpoint_event)


def _stream_pending_approval_events(
    state: Mapping[str, Any],
    *,
    manager: CheckpointManager,
    trace: TraceRecorder,
    latest_node: str | None,
) -> Iterator[dict[str, Any]]:
    runtime = state.get("runtime")
    tracker = getattr(runtime, "approval_tracker", None)
    if tracker is None:
        return
    for event in tracker.drain_events():
        trace.record_custom_event(event)
        if manager.mode == "strict":
            _save_and_trace_checkpoint(
                manager,
                trace,
                state,
                status="running",
                latest_node=latest_node,
                event=event,
                snapshot=False,
            )
        yield {"type": "custom_event", "event": event}


def _finish_run(
    manager: CheckpointManager,
    trace: TraceRecorder,
    state: Mapping[str, Any],
    *,
    status: str,
    latest_node: str | None,
    full_snapshot: bool,
) -> None:
    """Finalize checkpoint and trace independently and idempotently."""

    checkpoint_event: dict[str, Any] | None = None
    try:
        if full_snapshot:
            checkpoint_event = manager.save(
                state,
                status=status,
                latest_node=latest_node,
                snapshot=True,
            )
        else:
            checkpoint_event = manager.finalize(
                state,
                status=status,
                latest_node=latest_node,
            )
    except Exception as exc:
        trace.record_custom_event(
            {
                "type": "finalization_error",
                "component": "checkpoint",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    else:
        if checkpoint_event is not None:
            trace.record_custom_event(checkpoint_event)

    runtime = state.get("runtime")
    tracker = getattr(runtime, "approval_tracker", None)
    if tracker is not None:
        for event in tracker.drain_events():
            trace.record_custom_event(event)

    try:
        trace.end(
            status=status,
            latest_node=latest_node,
            final_state=state,
        )
    except Exception:
        # The checkpoint terminal marker has already been written. A trace
        # rendering failure must never revert it to running.
        pass
