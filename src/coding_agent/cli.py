from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated

import typer
from prompt_toolkit import PromptSession
from rich.console import Console

from coding_agent.application import AgentApplication
from coding_agent.events import (
    AgentCompleted,
    AgentFailed,
    TextDelta,
    ToolFinished,
    ToolStarted,
    WarningRaised,
)
from coding_agent.runtime import RuntimeConfigurationError, RuntimeSettings, create_runtime

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
    max_steps: Annotated[int, typer.Option(min=1, help="Maximum Agent steps per turn.")] = 20,
    max_tokens: Annotated[int, typer.Option(min=1, help="Maximum output tokens.")] = 4096,
    context_window: Annotated[int, typer.Option(min=2, help="Model context window.")] = 200_000,
    auto_compact_ratio: Annotated[
        float, typer.Option(min=0.01, max=0.99, help="Automatic compaction threshold ratio.")
    ] = 0.8,
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
) -> None:
    while True:
        prompt = (await read_input()).strip()
        if not prompt:
            continue
        if prompt == "/exit":
            return
        if prompt == "/help":
            write_output("Commands: /help, /context, /compact, /exit")
            continue
        if prompt == "/context":
            status = application.context_status()
            if status is None:
                write_output("Context management is unavailable.\n")
            else:
                write_output(
                    f"Context: {status.used_tokens}/{status.context_window} tokens "
                    f"estimated ({status.level}); auto={status.soft_limit}, "
                    f"hard={status.hard_limit}"
                )
                if status.last_provider_input_tokens is not None:
                    write_output(f"; last provider input={status.last_provider_input_tokens} exact")
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
                    f"Compacted context: {checkpoint.before_tokens} -> "
                    f"{checkpoint.after_tokens} tokens.\n"
                )
            continue
        async for event in application.run(prompt):
            if isinstance(event, TextDelta):
                write_output(event.text)
            elif isinstance(event, ToolStarted):
                write_output(f"\n[tool] {event.tool_name} started\n")
            elif isinstance(event, ToolFinished):
                state = "error" if event.is_error else "done"
                write_output(f"[tool] {event.tool_name} {state}\n")
            elif isinstance(event, AgentCompleted):
                write_output("\n")
            elif isinstance(event, AgentFailed):
                write_output(f"\n[error] {event.message}\n")
            elif isinstance(event, WarningRaised):
                write_output(f"\n[warning] {event.message}\n")


def write_console(console: Console, value: str) -> None:
    console.print(value, end="", markup=False, highlight=False)


async def _run_cli(
    settings: RuntimeSettings,
    *,
    resume: bool,
    prompt: str | None,
) -> None:
    console = Console()
    async with await create_runtime(settings, resume=resume) as runtime:
        if prompt is not None:
            prompts = iter((prompt, "/exit"))

            async def read_once() -> str:
                return next(prompts)

            await run_repl(
                runtime.application,
                read_input=read_once,
                write_output=lambda value: write_console(console, value),
            )
            return

        session: PromptSession[str] = PromptSession()

        async def read_interactive() -> str:
            return await session.prompt_async("coding-agent> ")

        console.print(
            f"codingAgent | model={settings.model} | workspace={settings.workspace} | "
            f"session={runtime.session_id}"
        )
        await run_repl(
            runtime.application,
            read_input=read_interactive,
            write_output=lambda value: write_console(console, value),
        )


def main() -> None:
    app(prog_name="coding-agent")
