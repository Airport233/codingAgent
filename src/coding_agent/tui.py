from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import ClassVar, Literal

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Collapsible, Footer, Label, OptionList, Static, TextArea
from textual.widgets.option_list import Option
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


@dataclass(frozen=True, slots=True)
class SlashCommand:
    name: str
    description: str


SLASH_COMMANDS = (
    SlashCommand("help", "Show available commands"),
    SlashCommand("model", "Choose a model"),
    SlashCommand("context", "Show context usage"),
    SlashCommand("compact", "Compact conversation context"),
    SlashCommand("thinking", "Toggle thinking details"),
    SlashCommand("clear", "Start a new empty session"),
    SlashCommand("exit", "Exit codingAgent"),
)


class PromptTextArea(TextArea):
    """Soft-wrapping prompt editor with explicit submit and newline actions."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "submit", "Submit", show=False, priority=True),
        Binding("shift+enter", "newline", "New line", show=False, priority=True),
        Binding("up", "completion_up", "Previous", show=False, priority=True),
        Binding("down", "completion_down", "Next", show=False, priority=True),
        Binding("escape", "completion_dismiss", "Dismiss", show=False, priority=True),
        Binding("tab", "completion_accept", "Complete", show=False, priority=True),
    ]

    completion_active = False

    class Submitted(Message):
        def __init__(self, text_area: PromptTextArea) -> None:
            super().__init__()
            self.text_area = text_area

        @property
        def control(self) -> PromptTextArea:
            return self.text_area

        @property
        def value(self) -> str:
            return self.text_area.text

    class CompletionAction(Message):
        def __init__(self, action: Literal["up", "down", "dismiss", "complete", "select"]):
            super().__init__()
            self.action = action

    class NavigationAction(Message):
        def __init__(self, direction: Literal["up", "down"]):
            super().__init__()
            self.direction = direction

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, value: str) -> None:
        self.load_text(value)
        lines = value.split("\n")
        self.move_cursor((len(lines) - 1, len(lines[-1])))

    def action_submit(self) -> None:
        if self.completion_active:
            self.post_message(self.CompletionAction("select"))
        else:
            self.post_message(self.Submitted(self))

    def action_newline(self) -> None:
        self.insert("\n")

    def action_completion_up(self) -> None:
        if self.completion_active:
            self.post_message(self.CompletionAction("up"))
        else:
            self.post_message(self.NavigationAction("up"))

    def action_completion_down(self) -> None:
        if self.completion_active:
            self.post_message(self.CompletionAction("down"))
        else:
            self.post_message(self.NavigationAction("down"))

    def action_completion_dismiss(self) -> None:
        if self.completion_active:
            self.post_message(self.CompletionAction("dismiss"))

    def action_completion_accept(self) -> None:
        if self.completion_active:
            self.post_message(self.CompletionAction("complete"))
        else:
            self.screen.focus_next()


class CodingAgentTui(App[None]):
    CSS_PATH = "tui.tcss"
    TITLE = "codingAgent"
    SUB_TITLE = "local coding agent"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c,super+c", "cancel_turn", "Cancel / copy", show=True, priority=True),
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
        self._completion_mode: Literal["commands", "models"] | None = None
        self._completion_matches: list[str] = []
        self._dismissed_completion_value: str | None = None
        self._prompt_history: list[str] = []
        self._history_index: int | None = None
        self._last_history_text: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("codingAgent", id="brand")
        yield VerticalScroll(id="conversation")
        with Horizontal(id="status-bar"):
            yield Label(self.model, id="status-model")
            yield Label(self.workspace, id="status-workspace")
            yield Label(self._context_label(), id="status-context")
            yield Label(f"session {self.session_id[:8]}", id="status-session")
        with Vertical(id="completion-popup"):
            yield Label("Commands", id="completion-title")
            yield OptionList(id="completion-options", markup=False)
        yield PromptTextArea(
            placeholder="Describe a task · Enter submit · Shift+Enter new line",
            soft_wrap=True,
            show_line_numbers=False,
            highlight_cursor_line=False,
            id="composer",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#completion-popup", Vertical).display = False
        self.query_one("#composer", PromptTextArea).focus()

    @on(PromptTextArea.Submitted, "#composer")
    async def submit_prompt(self, event: PromptTextArea.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return
        event.text_area.clear()
        if prompt.startswith("/"):
            await self._command(prompt)
            return
        if self._turn_worker is not None and self._turn_worker.is_running:
            await self._notice("A turn is already running. Press Ctrl+C to cancel it.", "warning")
            return
        await self._mount(Static(prompt, markup=False, classes="message user-message"))
        self._prompt_history.append(prompt)
        self._history_index = None
        self._last_history_text = None
        event.text_area.disabled = True
        self._reset_turn_widgets()
        self._turn_worker = self.run_worker(self._run_turn(prompt), group="agent", exclusive=True)

    @on(TextArea.Changed, "#composer")
    def resize_composer(self, event: TextArea.Changed) -> None:
        self.call_after_refresh(self._resize_composer, event.text_area)
        self._sync_completion(event.text_area.text)

    @on(PromptTextArea.CompletionAction)
    async def handle_completion_action(self, event: PromptTextArea.CompletionAction) -> None:
        choices = self.query_one("#completion-options", OptionList)
        if event.action == "up":
            choices.action_cursor_up()
        elif event.action == "down":
            choices.action_cursor_down()
        elif event.action == "dismiss":
            self._dismissed_completion_value = self.query_one("#composer", PromptTextArea).text
            self._hide_completion()
        elif event.action in {"complete", "select"}:
            await self._accept_completion(complete_only=event.action == "complete")

    @on(PromptTextArea.NavigationAction)
    def navigate_composer(self, event: PromptTextArea.NavigationAction) -> None:
        composer = self.query_one("#composer", PromptTextArea)
        history_active = (
            self._history_index is not None and composer.text == self._last_history_text
        )
        if composer.text and not history_active:
            self._history_index = None
            self._last_history_text = None
            if event.direction == "up":
                composer.move_cursor((0, 0))
            else:
                lines = composer.text.split("\n")
                composer.move_cursor((len(lines) - 1, len(lines[-1])))
            return
        if not self._prompt_history:
            return
        if event.direction == "up":
            if history_active:
                assert self._history_index is not None
                self._history_index = max(0, self._history_index - 1)
            else:
                self._history_index = len(self._prompt_history) - 1
        elif not history_active:
            self._history_index = None
            self._last_history_text = None
            return
        else:
            assert self._history_index is not None
            if self._history_index < len(self._prompt_history) - 1:
                self._history_index += 1
            else:
                self._history_index = None
                self._last_history_text = None
                composer.clear()
                return
        assert self._history_index is not None
        history_text = self._prompt_history[self._history_index]
        self._last_history_text = history_text
        composer.value = history_text

    @on(OptionList.OptionSelected, "#completion-options")
    async def select_completion(self, event: OptionList.OptionSelected) -> None:
        if event.option_id is not None:
            await self._accept_completion(option_id=event.option_id)

    def _resize_composer(self, composer: TextArea) -> None:
        composer.styles.height = max(3, min(composer.wrapped_document.height + 2, 8))

    def _sync_completion(self, value: str) -> None:
        if self._dismissed_completion_value == value:
            self._hide_completion()
            return
        self._dismissed_completion_value = None
        first_line = value.split("\n", 1)[0]
        if not first_line.startswith("/"):
            self._hide_completion()
            return
        if first_line.startswith("/model "):
            query = first_line.removeprefix("/model ").casefold()
            models = tuple(dict.fromkeys(self.available_models or (self.model,)))
            matches = [model for model in models if query in model.casefold()]
            self._show_completion("models", matches)
            return
        command_token = first_line[1:]
        if any(character.isspace() for character in command_token):
            self._hide_completion()
            return
        query = command_token.casefold()
        matches = [
            f"/{command.name}"
            for command in SLASH_COMMANDS
            if command.name.casefold().startswith(query)
        ]
        self._show_completion("commands", matches)

    def _show_completion(self, mode: Literal["commands", "models"], matches: list[str]) -> None:
        popup = self.query_one("#completion-popup", Vertical)
        title = self.query_one("#completion-title", Label)
        choices = self.query_one("#completion-options", OptionList)
        self._completion_mode = mode
        self._completion_matches = matches
        title.update("Commands" if mode == "commands" else "Choose model")
        choices.clear_options()
        if mode == "commands":
            descriptions = {f"/{item.name}": item.description for item in SLASH_COMMANDS}
            choices.add_options(
                Option(f"{item:<12} {descriptions[item]}", id=item) for item in matches
            )
        else:
            choices.add_options(Option(item, id=item) for item in matches)
        if not matches:
            choices.add_option(Option("No matches", disabled=True))
            choices.highlighted = None
        else:
            choices.highlighted = 0
        # Two border rows plus the title row sit outside the option viewport.
        popup.styles.height = min(max(len(matches), 1) + 3, 10)
        popup.display = True
        self.query_one("#composer", PromptTextArea).completion_active = True

    def _hide_completion(self) -> None:
        self._completion_mode = None
        self._completion_matches = []
        self.query_one("#completion-popup", Vertical).display = False
        self.query_one("#composer", PromptTextArea).completion_active = False

    async def _accept_completion(
        self, *, complete_only: bool = False, option_id: str | None = None
    ) -> None:
        choices = self.query_one("#completion-options", OptionList)
        selected = option_id
        if selected is None and choices.highlighted is not None and self._completion_matches:
            selected = self._completion_matches[choices.highlighted]
        if selected is None:
            return
        composer = self.query_one("#composer", PromptTextArea)
        if self._completion_mode == "commands":
            if selected == "/model":
                composer.value = "/model "
                self._sync_completion(composer.value)
            elif complete_only:
                composer.value = selected
                self._sync_completion(composer.value)
            else:
                composer.clear()
                self._hide_completion()
                await self._command(selected)
        elif self._completion_mode == "models":
            if complete_only:
                composer.value = f"/model {selected}"
                self._sync_completion(composer.value)
            else:
                composer.clear()
                self._hide_completion()
                await self._command(f"/model {selected}")
        composer.focus()

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
                    self._finish_assistant_segment()
                    await self._tool_started(event)
                elif isinstance(event, ToolFinished):
                    self._tool_finished(event)
                elif isinstance(event, ContextUsageChanged):
                    self.query_one("#status-context", Label).update(
                        f"context ~{event.used_tokens}/{event.context_window} · {event.level}"
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
            self.query_one("#composer", PromptTextArea).disabled = False
            self.query_one("#composer", PromptTextArea).focus()

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
            if status is None:
                message = "Context management is unavailable."
            else:
                message = (
                    f"Context estimate: {status.used_tokens}/{status.context_window} tokens "
                    f"({status.used_tokens / status.context_window:.1%}, {status.level}); "
                    f"auto={status.soft_limit}, hard={status.hard_limit}"
                )
                if status.last_provider_input_tokens is not None:
                    message += (
                        f"; last Provider input={status.last_provider_input_tokens} tokens exact"
                    )
                checkpoint = self.application.context_checkpoint()
                if checkpoint is not None:
                    message += (
                        f"\nLast compaction: {checkpoint.strategy} summary, replaced "
                        f"{checkpoint.retained_from} exchanges, retained "
                        f"{len(checkpoint.projected) - 1}; estimate "
                        f"{checkpoint.before_tokens} → {checkpoint.after_tokens}."
                    )
            await self._notice(message, "info")
        elif prompt == "/compact":
            try:
                checkpoint = await self.application.compact_context()
            except Exception:
                await self._notice("Context compaction failed; original context retained.", "error")
                return
            self._refresh_context_label()
            message = (
                "Context is too short to compact."
                if checkpoint is None
                else (
                    f"Compacted context with {checkpoint.strategy} summary: estimated "
                    f"{checkpoint.before_tokens} → {checkpoint.after_tokens} tokens; "
                    f"replaced {checkpoint.retained_from} exchanges, retained "
                    f"{len(checkpoint.projected) - 1}."
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
        return f"context ~{status.used_tokens}/{status.context_window} · {status.level}"

    def _refresh_context_label(self) -> None:
        self.query_one("#status-context", Label).update(self._context_label())

    def _reset_turn_widgets(self) -> None:
        self._assistant = None
        self._assistant_text = ""
        self._thinking = None
        self._thinking_text = ""
        self._tools = {}

    def _finish_assistant_segment(self) -> None:
        """Make later text render after the event that ended this segment."""
        self._assistant = None
        self._assistant_text = ""

    async def action_toggle_thinking(self) -> None:
        self.thinking_visible = not self.thinking_visible
        if self._thinking is not None:
            self._thinking[0].collapsed = not self.thinking_visible
        state = "shown" if self.thinking_visible else "hidden"
        await self._notice(f"Thinking details: {state}.", "info")

    async def action_cancel_turn(self) -> None:
        if self._turn_worker is not None and self._turn_worker.is_running:
            self._turn_worker.cancel()
            return
        selected_text = self.screen.get_selected_text()
        if isinstance(self.focused, TextArea) and self.focused.selected_text:
            selected_text = self.focused.selected_text
        if selected_text:
            self.copy_to_clipboard(selected_text)
            await _copy_to_macos_clipboard(selected_text)

    async def action_quit_agent(self) -> None:
        if self._turn_worker is not None and self._turn_worker.is_running:
            self._turn_worker.cancel()
        await self.application.close_session()
        self.exit()


def _tool_title(event: ToolStarted) -> str:
    if event.tool_name == "shell":
        command = event.arguments.get("command")
        cwd = event.arguments.get("cwd", ".")
        return f"shell [{cwd}] $ {command}" if isinstance(command, str) else "shell"
    path = event.arguments.get("path")
    return f"{event.tool_name} · {path}" if isinstance(path, str) else event.tool_name


async def _copy_to_macos_clipboard(text: str) -> None:
    """Use the native clipboard where OSC 52 is unsupported by Apple Terminal."""
    if sys.platform != "darwin":
        return
    try:
        process = await asyncio.create_subprocess_exec(
            "pbcopy",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(process.communicate(text.encode()), timeout=2)
    except (OSError, TimeoutError):
        return
