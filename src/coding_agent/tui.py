from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import ClassVar

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Collapsible, Footer, Input, Label, Static
from textual.worker import Worker

from coding_agent.application import AgentApplication
from coding_agent.events import (
    AgentCancelled,
    AgentCompleted,
    AgentFailed,
    ContextUsageChanged,
    TextDelta,
    ThinkingDelta,
    ThinkingFinished,
    ThinkingStarted,
    ToolFinished,
    ToolStarted,
    WarningRaised,
)
from coding_agent.runtime import RuntimeConfigurationError


@dataclass(frozen=True, slots=True)
class CliTransition:
    application: AgentApplication
    model: str
    session_id: str
    available_models: tuple[str, ...]


TransitionCallback = Callable[[str | None], Awaitable[CliTransition]]


class CodingAgentTui(App[None]):
    CSS_PATH = "tui.tcss"
    TITLE = "codingAgent"
    SUB_TITLE = "local coding agent"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "cancel_turn", "Cancel turn", show=True),
        Binding("ctrl+q", "quit_agent", "Quit", show=True),
        Binding("ctrl+t", "toggle_thinking", "Thinking", show=True),
    ]

    def __init__(
        self,
        application: AgentApplication,
        *,
        model: str,
        workspace: str,
        session_id: str,
        available_models: tuple[str, ...] = (),
        switch_model: TransitionCallback | None = None,
        clear_session: TransitionCallback | None = None,
    ) -> None:
        super().__init__()
        self.application = application
        self.model = model
        self.workspace = workspace
        self.session_id = session_id
        self.available_models = available_models
        self.switch_model = switch_model
        self.clear_session = clear_session
        self.thinking_visible = False
        self._turn_worker: Worker[None] | None = None
        self._assistant: Static | None = None
        self._assistant_text = ""
        self._thinking: tuple[Collapsible, Static] | None = None
        self._thinking_text = ""
        self._tools: dict[str, tuple[Collapsible, Static, str]] = {}

    def compose(self) -> ComposeResult:
        yield Static("codingAgent", id="brand")
        yield VerticalScroll(id="conversation")
        with Horizontal(id="status-bar"):
            yield Label(self.model, id="status-model")
            yield Label(self.workspace, id="status-workspace")
            yield Label(self._context_label(), id="status-context")
            yield Label(f"session {self.session_id[:8]}", id="status-session")
        yield Input(placeholder="Describe a coding task or type /help", id="composer")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#composer", Input).focus()

    @on(Input.Submitted, "#composer")
    async def submit_prompt(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return
        event.input.value = ""
        if prompt.startswith("/"):
            await self._command(prompt)
            return
        if self._turn_worker is not None and self._turn_worker.is_running:
            await self._notice("A turn is already running. Press Ctrl+C to cancel it.", "warning")
            return
        await self._mount(Static(prompt, markup=False, classes="message user-message"))
        event.input.disabled = True
        self._reset_turn_widgets()
        self._turn_worker = self.run_worker(self._run_turn(prompt), group="agent", exclusive=True)

    async def _run_turn(self, prompt: str) -> None:
        try:
            async for event in self.application.run(prompt):
                if isinstance(event, TextDelta):
                    if self._assistant is None:
                        self._assistant = Static(
                            "", markup=False, classes="message assistant-message"
                        )
                        await self._mount(self._assistant)
                    self._assistant_text += event.text
                    self._assistant.update(self._assistant_text)
                elif isinstance(event, ThinkingStarted):
                    body = Static("Working…", markup=False, classes="thinking-body")
                    panel = Collapsible(
                        body,
                        title="Thinking · working",
                        collapsed=not self.thinking_visible,
                        classes="thinking-card",
                    )
                    self._thinking = (panel, body)
                    await self._mount(panel)
                elif isinstance(event, ThinkingDelta):
                    self._thinking_text += event.text
                    if self._thinking is not None:
                        self._thinking[1].update(self._thinking_text or "Working…")
                elif isinstance(event, ThinkingFinished):
                    if self._thinking is not None:
                        self._thinking[0].title = "Thinking · complete"
                elif isinstance(event, ToolStarted):
                    await self._tool_started(event)
                elif isinstance(event, ToolFinished):
                    self._tool_finished(event)
                elif isinstance(event, ContextUsageChanged):
                    self.query_one("#status-context", Label).update(
                        f"context {event.used_tokens}/{event.context_window} · {event.level}"
                    )
                elif isinstance(event, AgentFailed):
                    await self._notice(event.message, "error")
                elif isinstance(event, (AgentCancelled, WarningRaised)):
                    await self._notice(event.message, "warning")
                elif isinstance(event, AgentCompleted) and self._assistant is None and event.text:
                    self._assistant = Static(
                        event.text, markup=False, classes="message assistant-message"
                    )
                    await self._mount(self._assistant)
        finally:
            self.query_one("#composer", Input).disabled = False
            self.query_one("#composer", Input).focus()

    async def _tool_started(self, event: ToolStarted) -> None:
        details = json.dumps(event.arguments, ensure_ascii=False, indent=2)
        body = Static(details, markup=False, classes="tool-body")
        panel = Collapsible(
            body,
            title=f"● {_tool_title(event)} · running",
            collapsed=True,
            classes="tool-card running",
        )
        self._tools[event.call_id] = (panel, body, _tool_title(event))
        await self._mount(panel)

    def _tool_finished(self, event: ToolFinished) -> None:
        item = self._tools.get(event.call_id)
        if item is None:
            return
        panel, body, title = item
        panel.title = f"{'✓' if not event.is_error else '✕'} {title} · {event.status}"
        panel.remove_class("running")
        panel.add_class("failed" if event.is_error else "succeeded")
        body.update(event.content or "(no output)")

    async def _command(self, prompt: str) -> None:
        if prompt == "/help":
            await self._notice(
                "/model [provider/model]  /context  /compact  /thinking  /clear  /exit",
                "info",
            )
        elif prompt == "/model":
            choices = ", ".join(self.available_models) or self.model
            await self._notice(f"Model: {self.model}\nAvailable: {choices}", "info")
        elif prompt.startswith("/model "):
            if self.switch_model is None:
                await self._notice("Model switching is unavailable.", "error")
                return
            try:
                transition = await self.switch_model(prompt.removeprefix("/model ").strip())
            except RuntimeConfigurationError as error:
                await self._notice(f"Model switch failed: {error}", "error")
                return
            except Exception:
                await self._notice("Model switch failed.", "error")
                return
            self._install_transition(transition)
            await self._notice(f"Model switched to {self.model}.", "info")
        elif prompt == "/clear":
            if self.clear_session is None:
                await self._notice("Starting a new session is unavailable.", "error")
                return
            try:
                transition = await self.clear_session(None)
            except Exception:
                await self._notice("Unable to start a new session.", "error")
                return
            self._install_transition(transition)
            await self.query_one("#conversation", VerticalScroll).remove_children()
            await self._notice("Started a new empty session.", "info")
        elif prompt == "/thinking":
            await self.action_toggle_thinking()
        elif prompt == "/context":
            status = self.application.context_status()
            message = (
                "Context management is unavailable."
                if status is None
                else f"Context: {status.used_tokens}/{status.context_window} tokens "
                f"({status.used_tokens / status.context_window:.1%}, {status.level})"
            )
            await self._notice(message, "info")
        elif prompt == "/compact":
            try:
                checkpoint = await self.application.compact_context()
            except Exception:
                await self._notice("Context compaction failed; original context retained.", "error")
                return
            message = (
                "Context is too short to compact."
                if checkpoint is None
                else (
                    f"Compacted context: {checkpoint.before_tokens} → "
                    f"{checkpoint.after_tokens} tokens."
                )
            )
            await self._notice(message, "info")
        elif prompt == "/exit":
            await self.action_quit_agent()
        else:
            await self._notice(
                f"Unknown command: {prompt.split(maxsplit=1)[0]}. Use /help.", "error"
            )

    def _install_transition(self, transition: CliTransition) -> None:
        self.application = transition.application
        self.model = transition.model
        self.session_id = transition.session_id
        self.available_models = transition.available_models
        self.query_one("#status-model", Label).update(self.model)
        self.query_one("#status-session", Label).update(f"session {self.session_id[:8]}")
        self.query_one("#status-context", Label).update(self._context_label())

    async def _mount(self, widget: Static | Collapsible) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        await conversation.mount(widget)
        conversation.scroll_end(animate=False)

    async def _notice(self, message: str, kind: str) -> None:
        await self._mount(Static(message, markup=False, classes=f"message notice {kind}"))

    def _context_label(self) -> str:
        status = self.application.context_status()
        if status is None:
            return "context unavailable"
        return f"context {status.used_tokens}/{status.context_window} · {status.level}"

    def _reset_turn_widgets(self) -> None:
        self._assistant = None
        self._assistant_text = ""
        self._thinking = None
        self._thinking_text = ""
        self._tools = {}

    async def action_toggle_thinking(self) -> None:
        self.thinking_visible = not self.thinking_visible
        if self._thinking is not None:
            self._thinking[0].collapsed = not self.thinking_visible
        state = "shown" if self.thinking_visible else "hidden"
        await self._notice(f"Thinking details: {state}.", "info")

    def action_cancel_turn(self) -> None:
        if self._turn_worker is not None and self._turn_worker.is_running:
            self._turn_worker.cancel()

    async def action_quit_agent(self) -> None:
        self.action_cancel_turn()
        await self.application.close_session()
        self.exit()


def _tool_title(event: ToolStarted) -> str:
    if event.tool_name == "shell":
        command = event.arguments.get("command")
        cwd = event.arguments.get("cwd", ".")
        return f"shell [{cwd}] $ {command}" if isinstance(command, str) else "shell"
    path = event.arguments.get("path")
    return f"{event.tool_name} · {path}" if isinstance(path, str) else event.tool_name
