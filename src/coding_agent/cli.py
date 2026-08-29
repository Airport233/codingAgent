from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console

from coding_agent.application import AgentApplication
from coding_agent.approval import ApprovalMode
from coding_agent.events import (
    AgentCancelled,
    AgentCompleted,
    AgentFailed,
    ApprovalRequested,
    ContextUsageChanged,
    TextDelta,
    ThinkingDelta,
    ThinkingFinished,
    ThinkingStarted,
    ToolFinished,
    ToolStarted,
    WarningRaised,
)
from coding_agent.runtime import (
    AgentRuntime,
    RuntimeConfigurationError,
    RuntimeSettings,
    create_runtime,
)
from coding_agent.sessions.jsonl import JsonlSessionRepository, SessionSummary
from coding_agent.tui import CliTransition, CodingAgentTui, format_slash_help

type TransitionCallback = Callable[[str | None], Awaitable[CliTransition]]

app = typer.Typer(
    add_completion=False,
    help="A small, self-contained coding agent.",
    invoke_without_command=True,
)


@app.callback()
def root(
    model: Annotated[str | None, typer.Option(help="Provider model ID.")] = None,
    workspace: Annotated[Path | None, typer.Option(help="Project working directory.")] = None,
    resume: Annotated[bool, typer.Option(help="Resume the latest project session.")] = False,
    max_steps: Annotated[
        int | None, typer.Option(min=1, help="Maximum Agent steps per turn.")
    ] = None,
    max_tokens: Annotated[int | None, typer.Option(min=1, help="Maximum output tokens.")] = None,
    context_window: Annotated[int | None, typer.Option(min=2, help="Model context window.")] = None,
    auto_compact_ratio: Annotated[
        float | None,
        typer.Option(min=0.01, max=0.99, help="Automatic compaction threshold ratio."),
    ] = None,
    approval_mode: Annotated[
        str | None, typer.Option(help="Shell approval mode: auto, ask, or deny.")
    ] = None,
    prompt: Annotated[str | None, typer.Option(help="Run one prompt and exit.")] = None,
) -> None:
    """Run codingAgent from the terminal."""
    selected_workspace = (workspace or Path.cwd()).resolve()
    try:
        settings = RuntimeSettings.load(
            workspace=selected_workspace,
            model=model,
            max_steps=max_steps,
            max_tokens=max_tokens,
            context_window=context_window,
            auto_compact_ratio=auto_compact_ratio,
            approval_mode=approval_mode,
        )
        asyncio.run(_run_cli(settings, resume=resume, prompt=prompt))
    except RuntimeConfigurationError as error:
        typer.echo(f"Configuration error: {error}", err=True)
        raise typer.Exit(2) from error
    except KeyboardInterrupt:
        typer.echo("Cancelled.", err=True)


async def run_repl(
    application: AgentApplication,
    *,
    read_input: Callable[[], Awaitable[str]],
    write_output: Callable[[str], None],
    model: str = "unknown",
    session_id: str = "unknown",
    available_models: tuple[str, ...] = (),
    switch_model: TransitionCallback | None = None,
    clear_session: TransitionCallback | None = None,
) -> None:
    thinking_visible = False
    while True:
        prompt = (await read_input()).strip()
        if not prompt:
            continue
        if prompt == "/exit":
            await application.close_session()
            return
        if prompt == "/help":
            write_output(format_slash_help() + "\n")
            continue
        if prompt == "/mode":
            current = application.approval_mode()
            write_output(f"Approval mode: {current or 'unavailable'}\n")
            continue
        if prompt.startswith("/mode "):
            target = prompt.removeprefix("/mode ").strip()
            if target not in {"auto", "ask", "deny"}:
                write_output("Usage: /mode auto|ask|deny\n")
                continue
            if not application.set_approval_mode(cast(ApprovalMode, target)):
                write_output("Approval mode cannot be changed for this session.\n")
                continue
            write_output(f"Approval mode switched to {target}.\n")
            continue
        if prompt == "/model":
            choices = ", ".join(available_models) or model
            write_output(f"Model: {model}; available: {choices}\n")
            continue
        if prompt.startswith("/model "):
            target = prompt.removeprefix("/model ").strip()
            if not target or switch_model is None:
                write_output("Model switching is unavailable.\n")
                continue
            try:
                transition = await switch_model(target)
            except RuntimeConfigurationError as error:
                write_output(f"[error] Model switch failed: {error}\n")
                continue
            except Exception:
                write_output("[error] Model switch failed.\n")
                continue
            application = transition.application
            model = transition.model
            session_id = transition.session_id
            available_models = transition.available_models
            write_output(f"Model switched to {model}; session={session_id}.\n")
            continue
        if prompt == "/clear":
            if clear_session is None:
                write_output("Starting a new session is unavailable.\n")
                continue
            try:
                transition = await clear_session(None)
            except Exception:
                write_output("[error] Unable to start a new session.\n")
                continue
            application = transition.application
            model = transition.model
            session_id = transition.session_id
            available_models = transition.available_models
            write_output(f"Started a new empty session: {session_id}.\n")
            continue
        if prompt == "/thinking":
            thinking_visible = not thinking_visible
            state = "shown" if thinking_visible else "hidden"
            write_output(f"Thinking details: {state}.\n")
            continue
        if prompt == "/resume":
            write_output("Session selection requires the interactive TUI.\n")
            continue
        if prompt == "/context":
            status = application.context_status()
            if status is None:
                write_output("Context management is unavailable.\n")
            else:
                write_output(
                    f"Context estimate: {status.used_tokens}/{status.context_window} tokens "
                    f"({status.used_tokens / status.context_window:.1%}, "
                    f"{status.level}); auto={status.soft_limit}, "
                    f"hard={status.hard_limit}"
                )
                if status.last_provider_input_tokens is not None:
                    write_output(f"; last provider input={status.last_provider_input_tokens} exact")
                if status.model_projection_active:
                    write_output("; model-switch projection active")
                checkpoint = application.context_checkpoint()
                if checkpoint is not None:
                    write_output(
                        f"; last compaction={checkpoint.strategy}, "
                        f"replaced {checkpoint.retained_from}, "
                        f"retained {len(checkpoint.projected) - 1}, "
                        f"estimate {checkpoint.before_tokens}->{checkpoint.after_tokens}"
                    )
                write_output("\n")
            continue
        if prompt == "/compact":
            try:
                checkpoint = await application.compact_context()
            except Exception:
                write_output("Context compaction failed; original context retained.\n")
                continue
            if checkpoint is None:
                write_output("Context is too short to compact.\n")
            else:
                write_output(
                    f"Compacted context with {checkpoint.strategy} summary: estimated "
                    f"{checkpoint.before_tokens} -> {checkpoint.after_tokens} tokens; "
                    f"replaced {checkpoint.retained_from} exchanges, retained "
                    f"{len(checkpoint.projected) - 1}.\n"
                )
            continue
        if prompt.startswith("/"):
            write_output(f"Unknown command: {prompt.split(maxsplit=1)[0]}. Use /help.\n")
            continue
        async for event in application.run(prompt):
            if isinstance(event, TextDelta):
                write_output(event.text)
            elif isinstance(event, ThinkingStarted):
                if thinking_visible:
                    write_output("\n[thinking] ")
                else:
                    write_output("\n[thinking] working...\n")
            elif isinstance(event, ThinkingDelta):
                if thinking_visible:
                    write_output(event.text)
            elif isinstance(event, ThinkingFinished):
                if thinking_visible:
                    write_output("\n[/thinking]\n")
                else:
                    write_output("[thinking] done\n")
            elif isinstance(event, ContextUsageChanged):
                if event.level != "safe":
                    write_output(
                        f"[context] {event.used_tokens}/{event.context_window} "
                        f"tokens ({event.level})\n"
                    )
            elif isinstance(event, ApprovalRequested):
                operation = event.arguments.get("command", event.arguments)
                guardian_line = (
                    f"[guardian] {event.guardian_note}\n" if event.guardian_note else ""
                )
                write_output(
                    f"[approval] {event.tool_name}: {operation}\n"
                    f"{guardian_line}"
                    "Allow? [y] once / [a] session / [n] deny: "
                )
                answer = (await read_input()).strip().casefold()
                decision = (
                    "allow_session"
                    if answer in {"a", "always"}
                    else "allow_once"
                    if answer in {"y", "yes"}
                    else "deny"
                )
                await application.resolve_approval(event.request_id, decision)
            elif isinstance(event, ToolStarted):
                write_output(f"\n[tool] {_format_tool_call(event)}\n")
            elif isinstance(event, ToolFinished):
                write_output(f"[tool] {event.tool_name} {event.status}")
                details = _format_tool_result(event)
                if details:
                    write_output(f"\n{details}")
                write_output("\n")
            elif isinstance(event, AgentCompleted):
                write_output("\n")
            elif isinstance(event, AgentFailed):
                write_output(f"\n[error] {event.message}\n")
            elif isinstance(event, AgentCancelled):
                write_output(f"\n[cancelled] {event.message}\n")
            elif isinstance(event, WarningRaised):
                write_output(f"\n[warning] {event.message}\n")


def _format_tool_call(event: ToolStarted) -> str:
    if event.tool_name == "shell":
        command = event.arguments.get("command")
        if not isinstance(command, str):
            return "shell"
        cwd = event.arguments.get("cwd", ".")
        timeout = event.arguments.get("timeout_seconds")
        suffix = f" (timeout={timeout}s)" if isinstance(timeout, (int, float)) else ""
        return f"shell [{cwd}] $ {command}{suffix}"
    path = event.arguments.get("path")
    if isinstance(path, str):
        if event.tool_name == "edit_file":
            start = event.arguments.get("start_line")
            end = event.arguments.get("end_line")
            return f"edit_file {path}:{start}-{end}"
        return f"{event.tool_name} {path}"
    query = event.arguments.get("query")
    if isinstance(query, str):
        return f'{event.tool_name} "{query}"'
    return event.tool_name


def _format_tool_result(event: ToolFinished, limit: int = 4_000) -> str:
    content = event.content.strip()
    if not content:
        return ""
    if len(content) <= limit:
        return content
    return f"{content[:limit]}\n[output truncated for display]"


def write_console(console: Console, value: str) -> None:
    console.print(value, end="", markup=False, highlight=False)


async def _run_cli(
    settings: RuntimeSettings,
    *,
    resume: bool,
    prompt: str | None,
) -> None:
    console = Console()
    runtime = await create_runtime(settings, resume=resume)
    current_settings = settings

    async def switch_model(target: str | None) -> CliTransition:
        nonlocal runtime, current_settings
        if target is None:
            raise RuntimeConfigurationError("A model ID is required")
        next_settings = RuntimeSettings.load(
            workspace=current_settings.workspace,
            model=target,
            data_root=current_settings.data_root,
            max_tokens=current_settings.max_tokens_override,
            max_steps=current_settings.max_steps_override,
            context_window=current_settings.context_window_override,
            auto_compact_ratio=current_settings.auto_compact_ratio_override,
            approval_mode=current_settings.approval_mode_override,
        )
        next_runtime = await create_runtime(
            next_settings,
            resume_session_id=(
                runtime.session_id if runtime.application.conversation_history() else None
            ),
        )
        await runtime.aclose()
        runtime = next_runtime
        current_settings = next_settings
        return _transition(runtime, current_settings)

    async def clear_session(_unused: str | None) -> CliTransition:
        nonlocal runtime
        await runtime.application.close_session()
        await runtime.aclose()
        runtime = await create_runtime(current_settings)
        return _transition(runtime, current_settings)

    async def list_sessions() -> tuple[SessionSummary, ...]:
        repository = JsonlSessionRepository(current_settings.data_root)
        return await repository.list_sessions(current_settings.workspace)

    async def resume_session(session_id: str) -> CliTransition:
        nonlocal runtime
        if session_id == runtime.session_id:
            return _transition(runtime, current_settings)
        next_runtime = await create_runtime(
            current_settings,
            resume_session_id=session_id,
        )
        await runtime.application.close_session()
        await runtime.aclose()
        runtime = next_runtime
        return _transition(runtime, current_settings)

    try:
        if prompt is not None:
            prompts = iter((prompt, "/exit"))

            async def read_once() -> str:
                return next(prompts)

            await run_repl(
                runtime.application,
                read_input=read_once,
                write_output=lambda value: write_console(console, value),
                model=settings.model_key,
                session_id=runtime.session_id,
                available_models=settings.available_models,
                switch_model=switch_model,
                clear_session=clear_session,
            )
            return

        tui = CodingAgentTui(
            runtime.application,
            model=current_settings.model_key,
            workspace=_display_workspace(current_settings.workspace),
            session_id=runtime.session_id,
            available_models=current_settings.available_models,
            version=_package_version(),
            permissions=current_settings.approval_mode,
            switch_model=switch_model,
            clear_session=clear_session,
            list_sessions=list_sessions,
            resume_session=resume_session,
        )
        await tui.run_async()
    finally:
        await runtime.aclose()


def _transition(runtime: AgentRuntime, settings: RuntimeSettings) -> CliTransition:
    return CliTransition(
        runtime.application,
        settings.model_key,
        runtime.session_id,
        settings.available_models,
    )


def _package_version() -> str:
    try:
        return version("coding-agent")
    except PackageNotFoundError:
        return "development"


def _display_workspace(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(Path.home())
    except ValueError:
        return str(resolved)
    return "~" if not relative.parts else f"~/{relative.as_posix()}"


def _status_line(settings: RuntimeSettings, application: AgentApplication, session_id: str) -> str:
    provider = settings.model_key.partition("/")[0] if "/" in settings.model_key else "default"
    status = application.context_status()
    context = (
        f"{status.used_tokens}/{status.context_window} "
        f"({status.used_tokens / status.context_window:.1%})"
        if status is not None
        else "unavailable"
    )
    return (
        f"codingAgent | provider={provider} | model={settings.model} | "
        f"workspace={settings.workspace} | context={context} | session={session_id}"
    )


def main() -> None:
    app(prog_name="coding-agent")
