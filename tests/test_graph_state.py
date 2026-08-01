from typing import Annotated, get_args, get_origin, get_type_hints

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from ddclaw.graph.state import (
    AgentHandoff,
    CompressionEvent,
    LayeredMemory,
    DDclawGraphState,
    SourceItem,
    TodoItem,
    VerificationResult,
)


def test_graph_state_fields_are_optional() -> None:
    assert DDclawGraphState.__required_keys__ == frozenset()
    assert DDclawGraphState.__optional_keys__ == {
        "task",
        "runtime",
        "messages",
        "plan_summary",
        "todos",
        "research_notes",
        "sources",
        "agent_handoffs",
        "code_agent_summary",
        "context_summary",
        "context_token_count",
        "context_token_limit",
        "context_should_compress",
        "context_next_node",
        "compression_events",
        "memory_snapshot",
        "history_summary",
        "acceptance_criteria",
        "verification_commands",
        "verification_results",
        "verification_checks",
        "verification_reason",
        "recommended_next_instruction",
        "passed",
        "attempts",
        "max_attempts",
        "last_actor_summary",
        "last_error",
        "final_answer",
        "intent_route",
        "intent_reason",
        "intent_confidence",
        "chat_response",
        "session_id",
        "session_turn",
        "session_context",
    }


def test_nested_state_items_have_required_fields() -> None:
    assert TodoItem.__required_keys__ == {
        "id",
        "content",
        "status",
        "note",
    }
    assert VerificationResult.__required_keys__ == {
        "command",
        "ok",
        "exit_code",
        "stdout",
        "stderr",
    }
    assert SourceItem.__required_keys__ == frozenset()
    assert AgentHandoff.__required_keys__ == frozenset()
    assert AgentHandoff.__optional_keys__ == {
        "from_agent",
        "to_agent",
        "instruction",
        "result",
    }
    assert CompressionEvent.__required_keys__ == frozenset()
    assert LayeredMemory.__required_keys__ == {
        "rules",
        "working_memory",
        "history_summary_store",
    }


def test_messages_field_uses_add_messages_reducer() -> None:
    message_hint = get_type_hints(
        DDclawGraphState,
        include_extras=True,
    )["messages"]

    assert get_origin(message_hint) is Annotated
    value_type, reducer = get_args(message_hint)
    assert value_type == list[BaseMessage]
    assert reducer is add_messages


def test_state_graph_merges_messages_instead_of_overwriting() -> None:
    builder = StateGraph(DDclawGraphState)
    builder.add_node(
        "respond",
        lambda state: {
            "messages": [
                AIMessage(content="done", id="assistant-message"),
            ]
        },
    )
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    graph = builder.compile()

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="start", id="human-message"),
            ]
        }
    )

    assert [message.content for message in result["messages"]] == [
        "start",
        "done",
    ]
