"""Core supervisor, legacy actor, and verifier nodes for ddclaw."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    messages_to_dict,
)
from langchain_core.tools import StructuredTool
from langgraph.config import get_stream_writer
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import BaseModel, Field

from ddclaw.agents.code_agent import run_code_agent
from ddclaw.agents.search_agent import run_search_agent
from ddclaw.core.state import RuntimeState
from ddclaw.graph.memory import (
    _short_text,
    _trim_handoffs,
    build_layered_memory,
    format_layered_memory_for_prompt,
    memory_event,
)
from ddclaw.graph.state import (
    AgentHandoff,
    DDclawGraphState,
    SourceItem,
    TodoItem,
    VerificationCheck,
    VerificationResult,
)
from ddclaw.prompts.stage2 import ACTOR_PROMPT
from ddclaw.prompts.stage3 import PLANNER_PROMPT, VERIFIER_PROMPT
from ddclaw.prompts.stage4 import CONTEXT_COMPRESSION_PROMPT
from ddclaw.prompts.stage5 import CHAT_RESPONDER_PROMPT, INTENT_ROUTER_PROMPT
from ddclaw.providers import create_model
from ddclaw.tools.execution import execute_tool
from ddclaw.tools.bash_tool import BashTool
from ddclaw.tools.registry import build_read_only_tools, build_tools
from ddclaw.tools.todo_tools import TodoUpdateTool, TodoWriteTool

_PLANNER_MAX_LOOPS = 10
_ACTOR_MAX_LOOPS = 10
_VERIFIER_MAX_LOOPS = 6
_VERIFICATION_TIMEOUT_SECONDS = 120
_MAX_CAPTURE_CHARS = 20_000
_CONTEXT_SUMMARY_LIMIT = 8_000


class VerificationCheckOutput(BaseModel):
    """Validated check returned by the verifier model."""

    name: str = Field(min_length=1)
    passed: bool
    detail: str


class VerificationDecision(BaseModel):
    """Validated final JSON decision returned by the verifier model."""

    passed: bool
    reason: str
    checks: list[VerificationCheckOutput] = Field(default_factory=list)
    recommended_next_instruction: str = ""


class SpecialistInstructionInput(BaseModel):
    """Instruction passed by the supervisor to one specialist agent."""

    instruction: str = Field(
        min_length=1,
        description="Focused task delegated to the specialist agent.",
    )


class IntentDecisionOutput(BaseModel):
    """Validated decision returned by the entry intent router."""

    route: Literal["chat", "workflow"]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


def intent_router_node(state: DDclawGraphState) -> dict[str, Any]:
    """Classify the latest input as lightweight chat or workspace workflow."""

    emit = _get_event_writer()
    memory = build_layered_memory(state, node="intent_router")
    emit(memory_event(memory, node="intent_router"))
    response = create_model().invoke(
        [
            SystemMessage(content=INTENT_ROUTER_PROMPT),
            HumanMessage(content=_entry_input(state, memory)),
        ]
    )

    try:
        payload = _parse_json_object(_content_to_text(response.content))
        decision = IntentDecisionOutput.model_validate(payload)
    except (TypeError, ValueError) as exc:
        return {
            "intent_route": "workflow",
            "intent_reason": f"Invalid intent response; using workflow: {exc}",
            "intent_confidence": 0.0,
        }

    route = decision.route
    if decision.confidence < 0.55:
        route = "workflow"
    return {
        "intent_route": route,
        "intent_reason": decision.reason,
        "intent_confidence": decision.confidence,
    }


def chat_responder_node(state: DDclawGraphState) -> dict[str, str]:
    """Answer conversational input directly without binding or calling tools."""

    emit = _get_event_writer()
    memory = build_layered_memory(state, node="chat_responder")
    emit(memory_event(memory, node="chat_responder"))
    response = create_model().invoke(
        [
            SystemMessage(content=CHAT_RESPONDER_PROMPT),
            HumanMessage(content=_entry_input(state, memory)),
        ]
    )
    chat_response = _content_to_text(response.content).strip()
    if not chat_response:
        chat_response = "How can I help?"
    return {
        "chat_response": chat_response,
        "final_answer": chat_response,
    }


def intent_route_fn(state: DDclawGraphState) -> str:
    """Route only an explicit chat decision to the lightweight responder."""

    return (
        "chat_responder"
        if state.get("intent_route") == "chat"
        else "planner"
    )


def planner_node(state: DDclawGraphState) -> dict[str, Any]:
    """Plan work and coordinate search/code specialists through tools."""

    runtime = _require_runtime(state)
    runtime.approval_tracker.set_attempt(state.get("attempts", 0) + 1)
    emit = _get_event_writer()
    working_state: dict[str, Any] = dict(state)
    working_state["messages"] = []
    working_state["agent_handoffs"] = [
        dict(item) for item in state.get("agent_handoffs", [])
    ]
    working_state["sources"] = [
        dict(item) for item in state.get("sources", [])
    ]

    todo_writer = TodoWriteTool()
    todo_write_tool = todo_writer.as_structured_tool()

    def call_search_agent(instruction: str) -> dict[str, Any]:
        return _call_search_agent_tool(
            working_state,
            emit,
            instruction,
        )

    def call_code_agent(instruction: str) -> dict[str, Any]:
        return _call_code_agent_tool(
            working_state,
            emit,
            instruction,
        )

    call_search_tool = StructuredTool.from_function(
        func=call_search_agent,
        name="call_search_agent",
        description=(
            "Delegate focused factual or document research to searchAgent."
        ),
        args_schema=SpecialistInstructionInput,
    )
    call_code_tool = StructuredTool.from_function(
        func=call_code_agent,
        name="call_code_agent",
        description=(
            "Delegate focused workspace file and code implementation to codeAgent."
        ),
        args_schema=SpecialistInstructionInput,
    )
    tools = [todo_write_tool, call_search_tool, call_code_tool]
    tools_by_name = {tool.name: tool for tool in tools}
    planner = create_model().bind_tools(tools)

    memory = build_layered_memory(working_state, node="planner")
    emit(memory_event(memory, node="planner"))
    planner_messages: list[BaseMessage] = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=_planner_input(working_state, memory)),
    ]

    for _ in range(_PLANNER_MAX_LOOPS):
        response = planner.invoke(planner_messages)
        planner_messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break

        for call in tool_calls:
            name = str(call.get("name", ""))
            args = call.get("args") or {}
            emit(
                {
                    "type": "tool_call",
                    "agent": "planner",
                    "name": name,
                    "args": args,
                }
            )
            result = execute_tool(call, tools_by_name=tools_by_name)
            if name == todo_write_tool.name and isinstance(result, Mapping):
                _apply_plan_result(working_state, result)

            visible_result = _planner_visible_result(result)
            planner_messages.append(
                ToolMessage(
                    content=json.dumps(
                        visible_result,
                        ensure_ascii=False,
                        default=str,
                    ),
                    tool_call_id=str(call.get("id", "")),
                )
            )
            emit(
                {
                    "type": "tool_result",
                    "agent": "planner",
                    "name": name,
                    "result": visible_result,
                }
            )

    update: dict[str, Any] = {
        "plan_summary": working_state.get("plan_summary", ""),
        "todos": working_state.get("todos", []),
        "acceptance_criteria": working_state.get("acceptance_criteria", []),
        "verification_commands": working_state.get("verification_commands", []),
        "research_notes": working_state.get("research_notes", ""),
        "sources": working_state.get("sources", []),
        "agent_handoffs": working_state.get("agent_handoffs", []),
        "code_agent_summary": working_state.get("code_agent_summary", ""),
        "context_next_node": "verifier",
    }
    delegated_messages = working_state.get("messages", [])
    if delegated_messages:
        update["messages"] = delegated_messages
    return update


def actor_node(state: DDclawGraphState) -> dict[str, Any]:
    """Execute the current plan and stream ReAct progress as custom events."""

    runtime = _require_runtime(state)
    runtime.approval_tracker.set_attempt(state.get("attempts", 0) + 1)
    todo_updater = TodoUpdateTool(state.get("todos", []))
    todo_update_tool = todo_updater.as_structured_tool()
    tools = [*build_tools(runtime), todo_update_tool]
    tools_by_name = {tool.name: tool for tool in tools}
    agent = create_model().bind_tools(tools)
    emit = _get_event_writer()

    request = {
        "task": state.get("task", ""),
        "plan_summary": state.get("plan_summary", ""),
        "todos": state.get("todos", []),
        "acceptance_criteria": state.get("acceptance_criteria", []),
        "verification_commands": state.get("verification_commands", []),
        "previous_verifier_error": state.get("last_error", ""),
    }
    actor_messages: list[BaseMessage] = [
        SystemMessage(content=ACTOR_PROMPT),
        HumanMessage(
            content=json.dumps(request, ensure_ascii=False, default=str),
        ),
    ]
    last_actor_summary = ""

    for _ in range(_ACTOR_MAX_LOOPS):
        response = agent.invoke(actor_messages)
        actor_messages.append(response)
        last_actor_summary = _content_to_text(response.content)
        emit(
            {
                "type": "ai_message",
                "content": last_actor_summary,
            }
        )

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break

        for call in tool_calls:
            name = str(call.get("name", ""))
            args = call.get("args") or {}
            emit(
                {
                    "type": "tool_call",
                    "name": name,
                    "args": args,
                }
            )
            result = execute_tool(call, tools_by_name=tools_by_name)
            actor_messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=str(call.get("id", "")),
                )
            )
            emit(
                {
                    "type": "tool_result",
                    "name": name,
                    "result": result,
                }
            )

    if not last_actor_summary:
        last_actor_summary = (
            f"Actor stopped after {_ACTOR_MAX_LOOPS} loops without a final summary."
        )
    emit(
        {
            "type": "final_answer",
            "content": last_actor_summary,
        }
    )
    return {
        "messages": actor_messages,
        "last_actor_summary": last_actor_summary,
        "todos": todo_updater.todos,
    }


def verifier_node(state: DDclawGraphState) -> dict[str, Any]:
    """Run deterministic commands and an independent read-only model review."""

    runtime = _require_runtime(state)
    runtime.approval_tracker.set_attempt(state.get("attempts", 0) + 1)
    verification_results = [
        _run_verification_command(command, runtime)
        for command in state.get("verification_commands", [])
        if command.strip()
    ]

    read_only_tools = build_read_only_tools(runtime)
    tools_by_name = {tool.name: tool for tool in read_only_tools}
    verifier_model = create_model()
    verifier = verifier_model.bind_tools(read_only_tools)
    verifier_state = {
        **state,
        "verification_results": verification_results,
    }
    memory = build_layered_memory(verifier_state, node="verifier")
    emit = _get_event_writer()
    emit(memory_event(memory, node="verifier"))
    messages: list[BaseMessage] = [
        SystemMessage(content=VERIFIER_PROMPT),
        HumanMessage(
            content=_verifier_input(
                verifier_state,
                verification_results,
                memory,
            )
        ),
    ]

    final_response: AIMessage | None = None
    for _ in range(_VERIFIER_MAX_LOOPS):
        response = verifier.invoke(messages)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            final_response = response
            break
        for call in tool_calls:
            result = execute_tool(call, tools_by_name=tools_by_name)
            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=str(call.get("id", "")),
                )
            )

    if final_response is None:
        final_response = verifier_model.invoke(
            [
                *messages,
                HumanMessage(
                    content=(
                        "The read-only tool budget is exhausted. Do not call "
                        "any more tools. Based on the command results and "
                        "evidence already present above, return the required "
                        "verifier JSON object now."
                    )
                ),
            ]
        )

    decision = _verification_decision(final_response)
    command_checks = [
        {
            "name": f"command: {result['command']}",
            "passed": result["ok"],
            "detail": _command_result_detail(result),
        }
        for result in verification_results
    ]
    verification_checks: list[VerificationCheck] = [
        check.model_dump() for check in decision.checks
    ]
    verification_checks.extend(command_checks)

    commands_passed = all(result["ok"] for result in verification_results)
    passed = decision.passed and commands_passed
    reason = decision.reason
    if not commands_passed:
        failed_commands = [
            result["command"] for result in verification_results if not result["ok"]
        ]
        reason = (
            f"{reason} Failed verification command(s): "
            f"{', '.join(failed_commands)}"
        ).strip()

    recommended = decision.recommended_next_instruction
    last_error = "" if passed else (recommended or reason)
    todos = _verified_todos(
        state.get("todos", []),
        passed=passed,
        failure_note=last_error,
    )
    attempts = state.get("attempts", 0) + 1
    max_attempts = state.get("max_attempts", 3)
    if passed or attempts >= max_attempts:
        context_next_node = "final"
    else:
        context_next_node = "planner"

    return {
        "passed": passed,
        "attempts": attempts,
        "verification_results": verification_results,
        "verification_checks": verification_checks,
        "verification_reason": reason,
        "recommended_next_instruction": recommended,
        "last_error": last_error,
        "todos": todos,
        "context_next_node": context_next_node,
    }


def context_monitor_node(state: DDclawGraphState) -> dict[str, Any]:
    """Estimate graph context size and decide whether compression is needed."""

    memory = build_layered_memory(state, node="context_monitor")
    memory_payload = HumanMessage(
        content=format_layered_memory_for_prompt(memory),
    )
    messages = [*state.get("messages", []), memory_payload]
    token_count = _estimate_message_tokens(messages)

    context_token_limit = state.get("context_token_limit", 400_000)
    return {
        "context_token_count": token_count,
        "context_should_compress": token_count > context_token_limit,
        "context_next_node": state.get("context_next_node", "verifier"),
    }


def context_compressor_node(state: DDclawGraphState) -> dict[str, Any]:
    """Compress raw graph messages and persist the resumable history summary."""

    runtime = _require_runtime(state)
    raw_messages = list(state.get("messages", []))
    memory = build_layered_memory(state, node="context_compressor")
    compression_request = {
        "messages": messages_to_dict(raw_messages),
        "layered_memory": memory,
    }
    model = create_model()
    response = model.invoke(
        [
            SystemMessage(content=CONTEXT_COMPRESSION_PROMPT),
            HumanMessage(
                content=json.dumps(
                    compression_request,
                    ensure_ascii=False,
                    default=str,
                )
            ),
        ]
    )
    summary = _compression_summary(response)
    compressed_message = AIMessage(content=summary)
    new_token_count = _estimate_message_tokens(
        [compressed_message],
        model=model,
    )

    history_path = runtime.resolve_path("HISTORY_SUMMARY.md")
    history_path.write_text(summary, encoding="utf-8")

    previous_events = list(state.get("compression_events", []))
    compression_event = {
        "node": "context_compressor",
        "reason": "Context exceeded the configured token limit.",
        "session_turn": state.get("session_turn", 0),
        "token_count_before": state.get("context_token_count", 0),
        "token_count_after": new_token_count,
        "summary": _short_text(summary, 500),
    }
    truncated_state = _truncated_context_fields(state)
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            compressed_message,
        ],
        "context_summary": summary,
        "context_token_count": new_token_count,
        "context_should_compress": False,
        **truncated_state,
        "history_summary": summary,
        "compression_events": [*previous_events, compression_event],
    }


def context_monitor_route(state: DDclawGraphState) -> str:
    """Route terminal state, oversized context, or the requested next node."""

    if state.get("passed"):
        return "final"
    if state.get("context_should_compress"):
        return "context_compressor"
    return state.get("context_next_node", "verifier")


def context_compressor_route(state: DDclawGraphState) -> str:
    """Resume at the destination saved before context compression."""

    return state.get("context_next_node", "verifier")


def verifier_route(state: DDclawGraphState) -> str:
    """Route successful or exhausted runs to finalization, else re-plan."""

    if state.get("passed", False):
        return "final"
    if state.get("attempts", 0) >= state.get("max_attempts", 3):
        return "final"
    return "planner"


def _planner_input(
    state: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> str:
    """Format supervisor state and layered memory for the planner model."""

    current_todos = state.get("todos", [])
    request = {
        "task": state.get("task", ""),
        "mode": "revise" if current_todos else "create",
        "previous_plan": state.get("plan_summary", ""),
        "previous_todos": current_todos,
        "previous_acceptance_criteria": state.get("acceptance_criteria", []),
        "previous_verification_commands": state.get(
            "verification_commands",
            [],
        ),
        "research_notes": state.get("research_notes", ""),
        "sources": state.get("sources", []),
        "previous_code_agent_summary": state.get("code_agent_summary", ""),
        "previous_handoffs": state.get("agent_handoffs", []),
        "last_error": state.get("last_error", ""),
    }
    return _input_with_layered_memory(request, memory)


def _verifier_input(
    state: Mapping[str, Any],
    verification_results: list[VerificationResult],
    memory: Mapping[str, Any],
) -> str:
    """Format verification evidence and layered memory for the reviewer."""

    request = {
        "task": state.get("task", ""),
        "plan_summary": state.get("plan_summary", ""),
        "todos": state.get("todos", []),
        "acceptance_criteria": state.get("acceptance_criteria", []),
        "verification_commands": state.get("verification_commands", []),
        "verification_results": verification_results,
        "code_agent_summary": state.get("code_agent_summary", ""),
    }
    return _input_with_layered_memory(request, memory)


def _input_with_layered_memory(
    request: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> str:
    return (
        f"{json.dumps(request, ensure_ascii=False, default=str)}"
        "\n\nLayered memory:\n"
        f"{format_layered_memory_for_prompt(memory)}"
    )


def _entry_input(
    state: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> str:
    """Format latest user input plus resumable session context."""

    request = {
        "latest_input": state.get("task", ""),
        "session_context": state.get("session_context", ""),
        "workflow_context": {
            "context_summary": state.get("context_summary", ""),
            "history_summary": state.get("history_summary", ""),
            "plan_summary": state.get("plan_summary", ""),
            "last_error": state.get("last_error", ""),
            "attempts": state.get("attempts", 0),
        },
    }
    return _input_with_layered_memory(request, memory)


def _call_search_agent_tool(
    state: dict[str, Any],
    writer: Callable[[dict[str, Any]], None],
    instruction: str,
) -> dict[str, Any]:
    """Delegate research and persist its notes, sources, and handoff."""

    writer(
        {
            "type": "handoff",
            "from": "planner",
            "to": "searchAgent",
            "instruction": instruction,
        }
    )
    result = run_search_agent(
        state,
        instruction,
        writer=writer,
    )
    summary = str(result.get("summary") or "")
    state["research_notes"] = _append_research_notes(
        str(state.get("research_notes") or ""),
        summary,
    )
    state["sources"] = _merge_source_items(
        state.get("sources", []),
        _source_items_from_search_result(result),
    )
    state["agent_handoffs"] = [
        *state.get("agent_handoffs", []),
        AgentHandoff(
            from_agent="planner",
            to_agent="searchAgent",
            instruction=instruction,
            result=summary,
        ),
    ]
    return result


def _call_code_agent_tool(
    state: dict[str, Any],
    writer: Callable[[dict[str, Any]], None],
    instruction: str,
) -> dict[str, Any]:
    """Delegate implementation and persist its todos, messages, and handoff."""

    writer(
        {
            "type": "handoff",
            "from": "planner",
            "to": "codeAgent",
            "instruction": instruction,
        }
    )
    result = run_code_agent(
        state,
        instruction,
        writer=writer,
    )
    summary = str(result.get("summary") or "")
    todos = result.get("todos")
    if isinstance(todos, list):
        state["todos"] = todos
    state["code_agent_summary"] = summary

    messages = result.get("messages")
    if isinstance(messages, list):
        state["messages"] = [
            *state.get("messages", []),
            *messages,
        ]
    state["agent_handoffs"] = [
        *state.get("agent_handoffs", []),
        AgentHandoff(
            from_agent="planner",
            to_agent="codeAgent",
            instruction=instruction,
            result=summary,
        ),
    ]
    return result


def _apply_plan_result(
    state: dict[str, Any],
    result: Mapping[str, Any],
) -> None:
    required_fields = (
        "plan_summary",
        "todos",
        "acceptance_criteria",
        "verification_commands",
    )
    if not all(field in result for field in required_fields):
        return
    for field in required_fields:
        state[field] = result[field]


def _planner_visible_result(result: Any) -> Any:
    if not isinstance(result, Mapping):
        return result
    return {
        key: value
        for key, value in result.items()
        if key not in {"messages", "tool_events"}
    }


def _append_research_notes(existing: str, new_summary: str) -> str:
    existing = existing.strip()
    new_summary = new_summary.strip()
    if not new_summary:
        return existing
    if not existing:
        return new_summary
    if new_summary in existing:
        return existing
    return f"{existing}\n\n{new_summary}"


def _source_items_from_search_result(
    result: Mapping[str, Any],
) -> list[SourceItem]:
    items: list[SourceItem] = []
    tool_events = result.get("tool_events", [])
    if isinstance(tool_events, list):
        for event in tool_events:
            if not isinstance(event, Mapping):
                continue
            if event.get("type") != "search_results":
                continue
            search_results = event.get("results", [])
            if not isinstance(search_results, list):
                continue
            for search_result in search_results:
                if isinstance(search_result, Mapping):
                    items.append(_normalize_source_item(search_result))

    sources = result.get("sources", [])
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, Mapping):
                items.append(_normalize_source_item(source))
            elif isinstance(source, str) and source:
                items.append(SourceItem(url=source))
    return _merge_source_items([], items)


def _normalize_source_item(source: Mapping[str, Any]) -> SourceItem:
    item: SourceItem = {}
    for field in ("title", "url", "content"):
        value = source.get(field)
        if isinstance(value, str):
            item[field] = value
    score = source.get("score")
    if isinstance(score, (int, float)):
        item["score"] = float(score)
    return item


def _merge_source_items(
    existing: list[SourceItem],
    new_items: list[SourceItem],
) -> list[SourceItem]:
    merged: list[SourceItem] = [dict(item) for item in existing]
    urls = {
        item.get("url")
        for item in merged
        if isinstance(item.get("url"), str) and item.get("url")
    }
    for item in new_items:
        url = item.get("url")
        if isinstance(url, str) and url:
            if url in urls:
                continue
            urls.add(url)
        elif item in merged:
            continue
        merged.append(dict(item))
    return merged


def _verification_decision(
    response: AIMessage | None,
) -> VerificationDecision:
    if response is None:
        return VerificationDecision(
            passed=False,
            reason=(
                f"Verifier did not return a final decision within "
                f"{_VERIFIER_MAX_LOOPS} loops."
            ),
            recommended_next_instruction=(
                "Inspect the verifier tool loop and provide a final JSON decision."
            ),
        )
    try:
        payload = _parse_json_object(_content_to_text(response.content))
        return VerificationDecision.model_validate(payload)
    except (ValueError, TypeError) as exc:
        return VerificationDecision(
            passed=False,
            reason=f"Verifier returned invalid JSON: {exc}",
            recommended_next_instruction=(
                "Re-run verification and return the required JSON object."
            ),
        )


def _run_verification_command(
    command: str,
    runtime: RuntimeState,
) -> VerificationResult:
    try:
        result = BashTool(runtime).run_bash(
            command,
            timeout_seconds=_VERIFICATION_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        return {
            "command": command,
            "ok": False,
            "exit_code": None,
            "stdout": "",
            "stderr": _bounded_text(str(exc)),
        }
    except OSError as exc:
        return {
            "command": command,
            "ok": False,
            "exit_code": None,
            "stdout": "",
            "stderr": str(exc),
        }

    stderr = _bounded_text(result.get("stderr"))
    error = _bounded_text(result.get("error"))
    if error:
        stderr = f"{stderr}\n{error}".strip()
    return {
        "command": command,
        "ok": bool(result.get("ok")),
        "exit_code": result.get("exit_code"),
        "stdout": _bounded_text(result.get("stdout")),
        "stderr": stderr,
    }


def _verified_todos(
    todos: list[TodoItem],
    *,
    passed: bool,
    failure_note: str,
) -> list[TodoItem]:
    updated = [dict(item) for item in todos]
    if passed:
        for item in updated:
            item["status"] = "completed"
            if not item["note"]:
                item["note"] = "Verified successfully."
        return updated

    blocked = False
    for item in updated:
        if item["status"] == "in_progress":
            item["status"] = "blocked"
            item["note"] = failure_note
            blocked = True
    if not blocked:
        candidates = [
            item
            for item in updated
            if any(
                marker in item.get("content", "").lower()
                for marker in ("verify", "verification", "test", "验证", "测试")
            )
        ]
        target = candidates[-1] if candidates else (updated[-1] if updated else None)
        if target is not None:
            target["status"] = "blocked"
            target["note"] = failure_note
    return updated


def _require_runtime(state: DDclawGraphState) -> RuntimeState:
    runtime = state.get("runtime")
    if runtime is None:
        raise ValueError("Graph state is missing runtime")
    return runtime


def _get_event_writer() -> Callable[[dict[str, Any]], None]:
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda event: None


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response does not contain a JSON object")
    parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response JSON must be an object")
    return parsed


def _compression_summary(response: BaseMessage) -> str:
    """Extract and bound the durable summary from compressor JSON output."""

    response_text = _content_to_text(response.content)
    try:
        payload = _parse_json_object(response_text)
    except (json.JSONDecodeError, ValueError, TypeError):
        summary = response_text.strip()
    else:
        summary_value = payload.get("summary", "")
        summary = (
            summary_value.strip()
            if isinstance(summary_value, str)
            else str(summary_value).strip()
        )
        if not summary:
            summary = json.dumps(payload, ensure_ascii=False, default=str)

    if not summary:
        summary = "No resumable context summary was returned by the compressor."
    return _short_text(summary, _CONTEXT_SUMMARY_LIMIT)


def _estimate_message_tokens(
    messages: list[BaseMessage],
    *,
    model: Any | None = None,
) -> int:
    """Use the provider tokenizer, falling back to a character estimate."""

    fallback_text = "\n".join(
        _content_to_text(message.content)
        for message in messages
    )
    try:
        counter = model if model is not None else create_model()
        raw_token_count = counter.get_num_tokens_from_messages(messages)
        if isinstance(raw_token_count, bool):
            raise TypeError("token count must be an integer")
        token_count = int(raw_token_count)
        if token_count < 0:
            raise ValueError("token count must not be negative")
        return token_count
    except Exception:
        return len(fallback_text) // 4


def _truncated_context_fields(state: DDclawGraphState) -> dict[str, Any]:
    """Bound large state fields retained after transcript compression."""

    handoffs = []
    for handoff in _trim_handoffs(state.get("agent_handoffs", [])):
        handoffs.append(
            {
                key: _short_text(value, 1_000)
                for key, value in handoff.items()
            }
        )

    sources: list[SourceItem] = []
    for source in state.get("sources", [])[-20:]:
        if not isinstance(source, Mapping):
            continue
        bounded_source: SourceItem = {}
        for key, limit in (
            ("title", 300),
            ("url", 1_000),
            ("content", 600),
        ):
            if key in source:
                bounded_source[key] = _short_text(source[key], limit)
        score = source.get("score")
        if isinstance(score, (int, float)):
            bounded_source["score"] = float(score)
        sources.append(bounded_source)

    todos: list[TodoItem] = []
    for todo in state.get("todos", []):
        todos.append(
            {
                "id": _short_text(todo.get("id", ""), 200),
                "content": _short_text(todo.get("content", ""), 800),
                "status": _short_text(todo.get("status", "pending"), 50),
                "note": _short_text(todo.get("note", ""), 800),
            }
        )

    verification_results: list[VerificationResult] = []
    for result in state.get("verification_results", [])[-10:]:
        verification_results.append(
            {
                "command": _short_text(result.get("command", ""), 1_000),
                "ok": bool(result.get("ok", False)),
                "exit_code": result.get("exit_code"),
                "stdout": _short_text(result.get("stdout", ""), 2_000),
                "stderr": _short_text(result.get("stderr", ""), 2_000),
            }
        )

    verification_checks: list[VerificationCheck] = []
    for check in state.get("verification_checks", [])[-20:]:
        verification_checks.append(
            {
                "name": _short_text(check.get("name", ""), 300),
                "passed": bool(check.get("passed", False)),
                "detail": _short_text(check.get("detail", ""), 1_000),
            }
        )

    return {
        "plan_summary": _short_text(state.get("plan_summary", ""), 1_600),
        "todos": todos,
        "acceptance_criteria": [
            _short_text(item, 800)
            for item in state.get("acceptance_criteria", [])
        ],
        "verification_commands": [
            _short_text(item, 1_000)
            for item in state.get("verification_commands", [])
        ],
        "research_notes": _short_text(
            state.get("research_notes", ""),
            1_600,
        ),
        "sources": sources,
        "agent_handoffs": handoffs,
        "code_agent_summary": _short_text(
            state.get("code_agent_summary", ""),
            1_000,
        ),
        "verification_results": verification_results,
        "verification_checks": verification_checks,
        "verification_reason": _short_text(
            state.get("verification_reason", ""),
            1_400,
        ),
        "recommended_next_instruction": _short_text(
            state.get("recommended_next_instruction", ""),
            1_400,
        ),
        "last_error": _short_text(state.get("last_error", ""), 1_400),
    }


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _command_result_detail(result: VerificationResult) -> str:
    output = result["stdout"] or result["stderr"]
    if output:
        return output
    return f"Command exited with code {result['exit_code']}."


def _bounded_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    if len(text) <= _MAX_CAPTURE_CHARS:
        return text
    return f"{text[:_MAX_CAPTURE_CHARS]}\n... output truncated ..."
