"""Shared state schema for the DDclaw LangGraph workflow."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from ddclaw.core.state import RuntimeState


class TodoItem(TypedDict):
    """One planned unit of work tracked by the graph."""

    id: str
    content: str
    status: str  # "pending" | "in_progress" | "completed" | "blocked"
    note: str


class VerificationResult(TypedDict):
    """Captured result from one verification command."""

    command: str
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str


class VerificationCheck(TypedDict):
    """One semantic acceptance check reported by the verifier model."""

    name: str
    passed: bool
    detail: str


class SourceItem(TypedDict, total=False):
    """One useful source collected by searchAgent."""

    title: str
    url: str
    content: str
    score: float


class AgentHandoff(TypedDict, total=False):
    """One supervisor delegation to a specialist agent."""

    from_agent: str
    to_agent: str
    instruction: str
    result: str


class CompressionEvent(TypedDict, total=False):
    """Metadata recorded when raw context is compressed."""

    node: str
    reason: str
    session_turn: int
    token_count_before: int
    token_count_after: int
    summary: str


class LayeredMemory(TypedDict):
    """Runtime snapshot containing the three memory layers."""

    rules: dict[str, Any]
    working_memory: dict[str, Any]
    history_summary_store: dict[str, Any]


class DDclawGraphState(TypedDict, total=False):
    """Shared, partially updatable state for the LangGraph workflow."""

    task: str
    runtime: RuntimeState
    messages: Annotated[list[BaseMessage], add_messages]
    plan_summary: str
    todos: list[TodoItem]
    research_notes: str
    sources: list[SourceItem]
    agent_handoffs: list[AgentHandoff]
    code_agent_summary: str
    context_summary: str
    context_token_count: int
    context_token_limit: int
    context_should_compress: bool
    context_next_node: str
    compression_events: list[CompressionEvent]
    memory_snapshot: LayeredMemory
    history_summary: str
    acceptance_criteria: list[str]
    verification_commands: list[str]
    verification_results: list[VerificationResult]
    verification_checks: list[VerificationCheck]
    verification_reason: str
    recommended_next_instruction: str
    passed: bool
    attempts: int
    max_attempts: int
    last_actor_summary: str
    last_error: str
    final_answer: str
    intent_route: str
    intent_reason: str
    intent_confidence: float
    chat_response: str
    session_id: str
    session_turn: int
    session_context: str
