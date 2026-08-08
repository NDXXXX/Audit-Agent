"""Compiled LangGraph entry and supervisor workflows for Audit Agent."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from audit_agent.graph.nodes import (
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
from audit_agent.graph.state import AuditGraphState
from audit_agent.prompts.stage2 import FINAL_PROMPT


def format_findings_summary(findings: list[dict[str, Any]] | None) -> str:
    """Build a one-line summary from a list of review findings."""

    if not findings:
        return ""
    by_dim: dict[str, int] = {}
    by_sev: dict[str, int] = {}
    by_verdict: dict[str, int] = {}
    for f in findings:
        if not isinstance(f, dict):
            continue
        dim = str(f.get("dimension", "?"))
        sev = str(f.get("severity", "medium")).lower()
        verdict = str(f.get("verdict", "unverified"))
        by_dim[dim] = by_dim.get(dim, 0) + 1
        by_sev[sev] = by_sev.get(sev, 0) + 1
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
    dims = ", ".join(f"{d}:{c}" for d, c in sorted(by_dim.items()))
    sevs = ", ".join(f"{s}:{c}" for s, c in sorted(by_sev.items()))
    verdicts = ", ".join(f"{v}:{c}" for v, c in sorted(by_verdict.items()))
    return (
        f"Audit: {len(findings)} findings"
        + (f" [{dims}]" if dims else "")
        + (f" ({sevs})" if sevs else "")
        + (f" [{verdicts}]" if verdicts else "")
    )


def final_node(state: AuditGraphState) -> dict[str, str]:
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
            format_findings_summary(state.get("verified_findings", []))
            or format_findings_summary(state.get("review_findings", []))
            or state.get("last_actor_summary", "")
        ),
        verification_reason=verification_reason,
        recommended_next_instruction=recommended,
    ).strip()
    return {"final_answer": final_answer}


def build_complex_workflow() -> Any:
    """Build the monitored supervisor workflow with context compression."""

    graph = StateGraph(AuditGraphState)
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

    graph = StateGraph(AuditGraphState)
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
