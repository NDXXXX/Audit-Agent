"""Typer entry point for the Audit Agent command."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text

from audit_agent.core.approval import (
    ApprovalDecision,
    ApprovalRequest,
    RunInterrupted,
)
from audit_agent.core.agent import stream_agent_events

app = typer.Typer(
    add_completion=False,
    help="Multi-agent code review with adversarial verification.",
    no_args_is_help=True,
)

console = Console()
error_console = Console(stderr=True)


@app.command()
def main(
    ctx: typer.Context,
    task: Annotated[
        str | None,
        typer.Argument(
            help="Review task for Audit Agent, optional when --resume is used."
        ),
    ] = None,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            "-w",
            help=(
                "Workspace directory. Defaults to .audit-workspace in the "
                "current directory."
            ),
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = None,
    max_attempts: Annotated[
        int,
        typer.Option(
            "--max-attempts",
            help="Maximum Planner/Supervisor → Verifier attempts.",
            min=1,
        ),
    ] = 3,
    approval_mode: Annotated[
        Literal["inline", "auto", "deny"],
        typer.Option(
            "--approval-mode",
            help="How risky BashTool commands are approved.",
        ),
    ] = "inline",
    checkpoint_mode: Annotated[
        Literal["light", "strict", "off"],
        typer.Option(
            "--checkpoint-mode",
            help="Checkpoint persistence level.",
        ),
    ] = "light",
    trace_mode: Annotated[
        Literal["on", "off"],
        typer.Option(
            "--trace-mode",
            help="Enable or disable execution tracing.",
        ),
    ] = "on",
    resume: Annotated[
        Path | None,
        typer.Option(
            "--resume",
            help="Resume from the checkpoint stored in this workspace.",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = None,
) -> None:
    """Run one task and render the workflow event stream in real time."""

    if task is None and resume is None:
        ctx.fail("Provide a task or use --resume <workspace>.")

    workspace_path = (
        resume
        or workspace
        or Path.cwd() / ".audit-workspace"
    )
    approval_handler = (
        _inline_approval_handler
        if approval_mode == "inline"
        else None
    )
    console.print(
        f"[bold cyan]🔍 Audit Agent run started[/bold cyan] · "
        f"workspace: {Path(workspace_path).expanduser().resolve()}"
    )
    console.print("[dim]🧠 Waiting for Planner and auditor events...[/dim]")
    try:
        for event in stream_agent_events(
            task or "",
            workspace=workspace_path,
            max_attempts=max_attempts,
            approval_mode=approval_mode,
            approval_handler=approval_handler,
            checkpoint_mode=checkpoint_mode,
            resume_workspace=resume,
            trace_mode=trace_mode,
        ):
            _render_event(event)
    except (RunInterrupted, KeyboardInterrupt):
        error_console.print()
        error_console.print(
            "[bold yellow]Run interrupted. Checkpoint finalization "
            "was requested.[/bold yellow]"
        )
        error_console.print(
            f"[dim]Resume with: audit --resume "
            f"{Path(workspace_path).expanduser().resolve()}[/dim]"
        )
        raise typer.Exit(code=130) from None
    except Exception as exc:
        error_console.print(
            f"[bold red]Error:[/bold red] {exc}",
        )
        raise typer.Exit(code=1) from exc


def _inline_approval_handler(
    request: ApprovalRequest,
) -> ApprovalDecision:
    console.print()
    console.print(
        Panel(
            Text(
                f"Risk: {request.risk_reason}\n\n"
                f"Command:\n{request.command}"
            ),
            title=f"⚠️ Approval required · {request.tool_name}",
            border_style="yellow",
        )
    )
    try:
        approved = typer.confirm("Approve this command?", default=False)
    except (typer.Abort, KeyboardInterrupt) as exc:
        raise RunInterrupted(
            f"Interrupted while deciding approval request {request.id}."
        ) from exc
    return ApprovalDecision(
        approved=approved,
        reason=(
            "Approved interactively by the user."
            if approved
            else "Rejected interactively by the user."
        ),
    )


def _render_event(event: dict[str, Any]) -> None:
    event_type = event.get("type")

    if event_type == "custom_event":
        _render_custom_event(event.get("event"))
        return

    if event_type == "graph_event":
        graph_event = event.get("event")
        if isinstance(graph_event, Mapping):
            for node, data in graph_event.items():
                _render_node_update(str(node), data)
        return

    if event_type == "custom":
        _render_custom_event(event.get("data"))
        return

    if event_type != "node_update":
        return

    _render_node_update(str(event.get("node", "")), event.get("data"))


def _render_node_update(node: str, data: Any) -> None:
    update = data if isinstance(data, Mapping) else {}

    if node == "planner":
        _render_planner(update)
    elif node == "actor":
        _render_actor(update)
    elif node == "verifier":
        _render_verifier(update)
    elif node == "final":
        content = str(
            update.get("final_answer") or "No final answer was returned."
        )
        console.print()
        console.print(
            Panel(
                Markdown(content),
                title="📝 Final",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                Pretty(data, expand_all=True),
                title=f"Workflow node · {node or 'unknown'}",
            )
        )


def _render_custom_event(data: Any) -> None:
    if not isinstance(data, Mapping):
        return

    custom_type = data.get("type")
    if custom_type == "handoff":
        console.print()
        console.print(
            Panel(
                Text(str(data.get("instruction", ""))),
                title=(
                    f"🤝 {data.get('from', 'planner')} → "
                    f"{data.get('to', 'specialist')}"
                ),
                border_style="magenta",
            )
        )
        return
    if custom_type == "approval_requested":
        console.print(
            f"[bold yellow]⚠ Approval request[/bold yellow] · "
            f"attempt {data.get('attempt', '?')} · "
            f"{data.get('risk_reason', '')}"
        )
        return
    if custom_type == "approval_decision":
        approved = bool(data.get("approved"))
        reused = " · reused" if data.get("reused") else ""
        console.print(
            f"[{'green' if approved else 'red'}]"
            f"{'✅ Approved' if approved else '⛔ Denied'}"
            f"[/]{reused} · attempt {data.get('attempt', '?')}"
        )
        return

    agent_name = str(data.get("agent") or "specialist")
    if custom_type == "tool_call":
        console.print()
        console.print(
            f"[bold cyan]🔧 {agent_name} · Tool call[/bold cyan] "
            f"[bold]{data.get('name', '')}[/bold]"
        )
        console.print(Pretty(data.get("args", {}), expand_all=True))
    elif custom_type == "tool_result":
        console.print(
            Panel(
                Text(_render_value(data.get("result"))),
                title=(
                    f"🔧 {agent_name} · Tool result · "
                    f"{data.get('name', '')}"
                ),
                border_style="cyan",
            )
        )
    elif custom_type == "search_results":
        console.print(
            Panel(
                Text(_render_value(dict(data))),
                title=f"🔎 {agent_name} · Search results",
                border_style="magenta",
            )
        )


def _render_planner(update: Mapping[str, Any]) -> None:
    parts = [
        str(update.get("plan_summary") or "No plan summary was returned."),
    ]
    todos = update.get("todos")
    if isinstance(todos, list) and todos:
        parts.append("\nTodos:")
        for item in todos:
            if isinstance(item, Mapping):
                status = item.get("status", "pending")
                content = item.get("content", "")
                parts.append(f"- [{status}] {content}")

    acceptance = update.get("acceptance_criteria")
    if isinstance(acceptance, list) and acceptance:
        parts.append("\nAcceptance criteria:")
        parts.extend(f"- {item}" for item in acceptance)

    commands = update.get("verification_commands")
    if isinstance(commands, list) and commands:
        parts.append("\nVerification commands:")
        parts.extend(f"- `{command}`" for command in commands)

    findings = update.get("verified_findings")
    if not isinstance(findings, list) or not findings:
        findings = update.get("review_findings")
    if isinstance(findings, list) and findings:
        parts.append(f"\nReview findings ({len(findings)}):")
        sev_styles = {
            "critical": "bold red",
            "high": "red",
            "medium": "yellow",
            "low": "dim",
        }
        verdict_marks = {"confirmed": "✓", "false_positive": "✗", "duplicate": "≫"}
        for f in findings:
            if not isinstance(f, Mapping):
                continue
            sev = str(f.get("severity", "medium")).lower()
            style = sev_styles.get(sev, "")
            verdict = str(f.get("verdict", ""))
            mark = verdict_marks.get(verdict, "")
            parts.append(
                f"- {mark} [{sev.upper()}] {f.get('dimension', '?')}: "
                f"{f.get('title', '?')} ({f.get('file', '?')}:{f.get('line', '?')})"
            )

    sources = update.get("sources")
    if isinstance(sources, list) and sources:
        parts.append("\nResearch sources:")
        for source in sources:
            if isinstance(source, Mapping):
                url = source.get("url", "")
                title = source.get("title") or url
                parts.append(f"- [{title}]({url})" if url else f"- {title}")

    handoffs = update.get("agent_handoffs")
    if isinstance(handoffs, list) and handoffs:
        parts.append("\nHandoffs:")
        for handoff in handoffs:
            if isinstance(handoff, Mapping):
                parts.append(
                    f"- {handoff.get('from_agent', 'planner')} → "
                    f"{handoff.get('to_agent', 'specialist')}: "
                    f"{handoff.get('instruction', '')}"
                )

    console.print()
    console.print(
        Panel(
            Markdown("\n".join(parts)),
            title="📋 Planner",
            border_style="blue",
        )
    )


def _render_actor(update: Mapping[str, Any]) -> None:
    summary = str(
        update.get("last_actor_summary") or "Actor returned no summary."
    )
    todos = update.get("todos")
    parts = [summary]
    if isinstance(todos, list) and todos:
        parts.append("\nTodo status:")
        for item in todos:
            if isinstance(item, Mapping):
                parts.append(
                    f"- [{item.get('status', 'pending')}] "
                    f"{item.get('content', '')}"
                )

    console.print()
    console.print(
        Panel(
            Markdown("\n".join(parts)),
            title="🔧 Actor",
            border_style="cyan",
        )
    )


def _render_verifier(update: Mapping[str, Any]) -> None:
    passed = bool(update.get("passed", False))
    icon = "✅" if passed else "❌"
    parts = [
        f"Attempt: {update.get('attempts', '?')}",
        str(
            update.get("verification_reason")
            or update.get("last_error")
            or "No verification reason was returned."
        ),
    ]

    checks = update.get("verification_checks")
    if isinstance(checks, list) and checks:
        parts.append("\nChecks:")
        for check in checks:
            if isinstance(check, Mapping):
                check_icon = "✅" if check.get("passed") else "❌"
                parts.append(
                    f"- {check_icon} {check.get('name', 'check')}: "
                    f"{check.get('detail', '')}"
                )

    results = update.get("verification_results")
    if isinstance(results, list) and results:
        parts.append("\nCommands:")
        for result in results:
            if isinstance(result, Mapping):
                result_icon = "✅" if result.get("ok") else "❌"
                parts.append(
                    f"- {result_icon} `{result.get('command', '')}` "
                    f"(exit {result.get('exit_code')})"
                )

    console.print()
    console.print(
        Panel(
            Markdown("\n".join(parts)),
            title=f"{icon} Verifier",
            border_style="green" if passed else "red",
        )
    )


def _render_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
