"""Compiled LangGraph entry and Stage 5 supervisor workflows for ddclaw."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ddclaw.graph.nodes import (
    context_compressor_node,
    context_compressor_route,
    context_monitor_node,
    context_monitor_route,
    chat_responder_node,
    intent_route_fn,
    intent_router_node,
    planner_node,
    verifier_node,
)
from ddclaw.graph.state import DDclawGraphState
from ddclaw.prompts.stage2 import FINAL_PROMPT


def final_node(state: DDclawGraphState) -> dict[str, str]:
    """Format the terminal workflow state as a user-facing final answer."""

    passed = state.get("passed", False)
    verification_reason = state.get("verification_reason") or state.get(
        "last_error",
        "",
    )
    if not verification_reason:
        verification_reason = (
            "All verification checks passed."
            if passed
            else "Verification did not pass."
        )

    recommended = state.get("recommended_next_instruction", "")
    if passed:
        recommended = "None — the task passed verification."
    elif not recommended:
        recommended = state.get("last_error") or (
            "Review the failed checks before continuing."
        )

    final_answer = FINAL_PROMPT.format(
        status="PASSED" if passed else "FAILED",
        task=state.get("task", ""),
        attempts=state.get("attempts", 0),
        max_attempts=state.get("max_attempts", 3),
        last_actor_summary=(
            state.get("code_agent_summary")
            or state.get("last_actor_summary", "")
        ),
        verification_reason=verification_reason,
        recommended_next_instruction=recommended,
    ).strip()
    return {"final_answer": final_answer}


def build_complex_workflow() -> Any:
    """Build the monitored supervisor workflow with context compression."""

    graph = StateGraph(DDclawGraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("context_monitor", context_monitor_node)
    graph.add_node("context_compressor", context_compressor_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "context_monitor")
    graph.add_conditional_edges(
        "context_monitor",
        context_monitor_route,
        {
            "context_compressor": "context_compressor",
            "verifier": "verifier",
            "planner": "planner",
            "final": "final",
        },
    )
    graph.add_conditional_edges(
        "context_compressor",
        context_compressor_route,
        {
            "verifier": "verifier",
            "planner": "planner",
            "final": "final",
        },
    )
    graph.add_edge("verifier", "context_monitor")
    graph.add_edge("final", END)
    return graph.compile()


def build_entry_workflow() -> Any:
    """Build the intent graph that separates chat from workspace tasks."""

    graph = StateGraph(DDclawGraphState)
    graph.add_node("intent_router", intent_router_node)
    graph.add_node("chat_responder", chat_responder_node)

    graph.add_edge(START, "intent_router")
    graph.add_conditional_edges(
        "intent_router",
        intent_route_fn,
        {
            "chat_responder": "chat_responder",
            "planner": END,
        },
    )
    graph.add_edge("chat_responder", END)
    return graph.compile()


def build_workflow() -> Any:
    """Build the current workflow while preserving the original public API."""

    return build_complex_workflow()
