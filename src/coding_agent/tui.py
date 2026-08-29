from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Literal, cast

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.timer import Timer
from textual.widgets import Collapsible, Footer, Label, OptionList, Static, TextArea
from textual.widgets.option_list import Option
from textual.worker import Worker

from coding_agent.application import AgentApplication
from coding_agent.approval import ApprovalDecision, ApprovalMode
from coding_agent.domain import (
    AssistantExchange,
    CompactionRecord,
    ConversationExchange,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolContinuationExchange,
    ToolResultBlock,
    ToolUseBlock,
    UnknownProviderBlock,
    UserExchange,
)
from coding_agent.events import (
    AgentCancelled,
    AgentCompleted,
    AgentFailed,
    AgentStarted,
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
from coding_agent.runtime import RuntimeConfigurationError
from coding_agent.sessions.jsonl import SessionSummary
from coding_agent.skills import format_skill_list
from coding_agent.skills.installer import InstallResult, SkillInstaller


@dataclass(frozen=True, slots=True)
class CliTransition:
    application: AgentApplication
    model: str
    session_id: str
    available_models: tuple[str, ...]


TransitionCallback = Callable[[str | None], Awaitable[CliTransition]]
SessionListCallback = Callable[[], Awaitable[tuple[SessionSummary, ...]]]
ResumeCallback = Callable[[str], Awaitable[CliTransition]]


@dataclass(frozen=True, slots=True)
class SlashCommand:
    name: str
    description: str
    arguments: str = ""


SLASH_COMMANDS = (
    SlashCommand("help", "Show available commands"),
    SlashCommand("model", "Show or choose a model", "[provider/model]"),
    SlashCommand("mode", "Show or set approval mode", "[auto|ask|deny]"),
    SlashCommand("skills", "List available coding workflows"),
    SlashCommand("skill", "Run a task with a coding workflow", "<name> <task>"),
    SlashCommand("context", "Show context usage"),
    SlashCommand("compact", "Compact conversation context"),
    SlashCommand("thinking", "Toggle thinking details"),
    SlashCommand("resume", "Resume a saved session"),
    SlashCommand("clear", "Start a new empty session"),
    SlashCommand("exit", "Exit codingAgent"),
)


def format_slash_help() -> str:
    usages = [
        f"/{command.name}{f' {command.arguments}' if command.arguments else ''}"
        for command in SLASH_COMMANDS
    ]
    width = max(map(len, usages))
    lines = ["Commands:"]
    lines.extend(
        f"  {usage:<{width}}  {command.description}"
        for usage, command in zip(usages, SLASH_COMMANDS, strict=True)
    )
    return "\n".join(lines)


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
        self.resize_to_content()

    def resize_to_content(self) -> None:
        self.styles.height = max(3, min(self.wrapped_document.height + 2, 8))

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


class CompactionProgress(Static):
    """Indeterminate progress display for a provider operation with no percentage."""

    FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self) -> None:
        super().__init__("", markup=False, classes="message notice compaction-progress running")
        self._started = time.monotonic()
        self._frame = 0
        self._timer: Timer | None = None

    def on_mount(self) -> None:
        self._tick()
        self._timer = self.set_interval(0.1, self._tick)

    def _tick(self) -> None:
        elapsed = int(time.monotonic() - self._started)
        self.update(f"{self.FRAMES[self._frame]} Compacting context… {elapsed}s")
        self._frame = (self._frame + 1) % len(self.FRAMES)

    def finish(self, message: str, kind: str = "info") -> None:
        if self._timer is not None:
            self._timer.pause()
        self.remove_class("running")
        self.add_class(kind)
        self.update(message)


class ResumeSessionScreen(Screen[str | None]):
    """Full-screen chooser for project-scoped saved sessions."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Back", show=True),
        Binding("ctrl+c", "cancel", "Back", show=False),
    ]

    def __init__(self, sessions: tuple[SessionSummary, ...], *, current_session_id: str) -> None:
        super().__init__()
        self.sessions = sessions
        self.current_session_id = current_session_id

    def compose(self) -> ComposeResult:
        yield Static("Resume a previous session", id="resume-title")
        options = [
            Option(self._option_label(session), id=session.session_id) for session in self.sessions
        ]
        if not options:
            options = [Option("No saved sessions for this workspace", disabled=True)]
        yield OptionList(*options, id="resume-options", markup=False)
        yield Static(self._preview(0), id="resume-preview", markup=False)
        yield Static("↑/↓ select · Enter resume · Esc cancel", id="resume-help")

    def on_mount(self) -> None:
        choices = self.query_one("#resume-options", OptionList)
        if self.sessions:
            choices.highlighted = 0
        choices.focus()

    @on(OptionList.OptionHighlighted, "#resume-options")
    def show_preview(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_id is None:
            return
        index = next(
            (
                index
                for index, session in enumerate(self.sessions)
                if session.session_id == event.option_id
            ),
            None,
        )
        if index is not None:
            self.query_one("#resume-preview", Static).update(self._preview(index))

    @on(OptionList.OptionSelected, "#resume-options")
    def select_session(self, event: OptionList.OptionSelected) -> None:
        if event.option_id is not None:
            self.dismiss(str(event.option_id))

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _option_label(self, session: SessionSummary) -> str:
        current = " · current" if session.session_id == self.current_session_id else ""
        compacted = " · compacted" if session.compacted else ""
        return f"{session.title}  ·  {_display_time(session.updated_at)}{compacted}{current}"

    def _preview(self, index: int) -> str:
        if not self.sessions:
            return "Start a conversation before using /resume."
        session = self.sessions[index]
        model = session.model or "unknown model"
        compacted = "yes" if session.compacted else "no"
        return (
            f"{session.title}\n\n"
            f"Session: {session.session_id}\n"
            f"Created: {_display_time(session.created_at)}\n"
            f"Updated: {_display_time(session.updated_at)}\n"
            f"Model: {model}\n"
            f"Exchanges: {session.exchange_count} · Compacted: {compacted}"
        )


class ApprovalScreen(ModalScreen[tuple[str, ApprovalDecision]]):
    """Small dialog over the dimmed conversation for a tool call needing approval.

    Deliberately not a full-screen takeover: the transcript stays visible
    behind the dialog, and the tool details panel is bounded rather than
    stretched to fill the terminal.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape,ctrl+c", "deny", "Deny", show=True),
    ]

    def __init__(self, event: ApprovalRequested) -> None:
        super().__init__()
        self.request = event

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Static("Permission required", id="approval-title")
            title = _tool_title(
                ToolStarted(self.request.call_id, self.request.tool_name, self.request.arguments)
            )
            details = f"{title}\n\n{_tool_arguments(self.request.arguments)}"
            if self.request.guardian_note:
                details += f"\n\n[Guardian] {self.request.guardian_note}"
            with VerticalScroll(id="approval-details"):
                yield Static(details, markup=False, id="approval-details-text")
            yield OptionList(
                Option("Allow once", id="allow_once"),
                Option("Always allow this tool for this session", id="allow_session"),
                Option("Deny", id="deny"),
                id="approval-options",
                markup=False,
            )
            yield Static("↑/↓ select · Enter confirm · Esc deny", id="approval-help")

    def on_mount(self) -> None:
        choices = self.query_one("#approval-options", OptionList)
        choices.highlighted = 0
        choices.focus()

    def on_resize(self, event: object) -> None:
        """Keep the dialog bounded inside the terminal as the window resizes.

        Textual's CSS ``max-height: 90%`` is not reliably honored on an
        auto-height container, so the cap is computed against the live screen
        size instead. The approval buttons (#approval-options) are always kept
        fully visible; the details panel absorbs whatever remains.
        """
        del event
        dialog = self.query_one("#approval-dialog", Vertical)
        # Leave a 2-cell margin top/bottom; never smaller than the three
        # fixed rows (title + options + help) plus padding.
        cap = max(7, self.size.height - 2)
        dialog.styles.max_height = cap
        details = self.query_one("#approval-details", VerticalScroll)
        details.styles.max_height = max(2, cap - 8)

    @on(OptionList.OptionSelected, "#approval-options")
    def select_decision(self, event: OptionList.OptionSelected) -> None:
        if event.option_id in {"allow_once", "allow_session", "deny"}:
            self.dismiss((self.request.request_id, cast(ApprovalDecision, event.option_id)))

    def action_deny(self) -> None:
        self.dismiss((self.request.request_id, "deny"))


_MODE_CYCLE: tuple[ApprovalMode, ...] = ("auto", "ask", "deny")


class CodingAgentTui(App[None]):
    CSS_PATH = "tui.tcss"
    TITLE = "codingAgent"
    SUB_TITLE = "local coding agent"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c,super+c", "cancel_turn", "Cancel / copy", show=True, priority=True),
        Binding("ctrl+q", "quit_agent", "Quit", show=True),
        Binding("ctrl+t", "toggle_thinking", "Thinking", show=True),
        Binding("shift+tab", "cycle_approval_mode", "Mode", show=True, priority=True),
    ]

    def __init__(
        self,
        application: AgentApplication,
        *,
        model: str,
        workspace: str,
        session_id: str,
        available_models: tuple[str, ...] = (),
        version: str = "unknown",
        permissions: str = "automatic",
        switch_model: TransitionCallback | None = None,
        clear_session: TransitionCallback | None = None,
        list_sessions: SessionListCallback | None = None,
        resume_session: ResumeCallback | None = None,
    ) -> None:
        super().__init__()
        self.application = application
        self.model = model
        self.workspace = workspace
        self.session_id = session_id
        self.available_models = available_models
        self.version = version
        self.permissions = permissions
        self.switch_model = switch_model
        self.clear_session = clear_session
        self.list_sessions = list_sessions
        self.resume_session = resume_session
        self.thinking_visible = False
        self._turn_worker: Worker[None] | None = None
        self._assistant: Static | None = None
        self._assistant_text = ""
        self._thinking: tuple[Collapsible, Static] | None = None
        self._thinking_text = ""
        self._tools: dict[str, tuple[Collapsible, Static, str, str]] = {}
        self._completion_mode: Literal["commands", "models", "skills"] | None = None
        self._completion_matches: list[str] = []
        self._dismissed_completion_value: str | None = None
        self._prompt_history: list[str] = []
        self._history_index: int | None = None
        self._last_history_text: str | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="conversation"):
            yield Static(self._welcome_text(), markup=False, id="brand")
        with Horizontal(id="status-bar"):
            yield Label(self._mode_label(), id="status-mode")
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

    async def on_mount(self) -> None:
        self.query_one("#completion-popup", Vertical).display = False
        await self._render_recovered_history()
        self._refresh_session_metadata()
        self._update_mode_indicator()
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
        await self._start_turn(prompt)

    async def _start_turn(
        self,
        prompt: str,
        *,
        skill_name: str | None = None,
        history_prompt: str | None = None,
    ) -> None:
        if self._turn_worker is not None and self._turn_worker.is_running:
            await self._notice("A turn is already running. Press Ctrl+C to cancel it.", "warning")
            return
        await self._mount(Static(prompt, markup=False, classes="message user-message"))
        self._prompt_history.append(history_prompt or prompt)
        self._history_index = None
        self._last_history_text = None
        self.query_one("#composer", PromptTextArea).disabled = True
        self._reset_turn_widgets()
        self._turn_worker = self.run_worker(
            self._run_turn(prompt, skill_name=skill_name), group="agent", exclusive=True
        )

    @on(TextArea.Changed, "#composer")
    def resize_composer(self, event: TextArea.Changed) -> None:
        composer = cast(PromptTextArea, event.text_area)
        self.call_after_refresh(self._resize_composer, composer)
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

    def _resize_composer(self, composer: PromptTextArea) -> None:
        composer.resize_to_content()

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
        if first_line.startswith("/skill "):
            remainder = first_line.removeprefix("/skill ")
            if " " in remainder:
                self._hide_completion()
                return
            query = remainder.casefold()
            matches = [
                name
                for name, _description, _source in self.application.available_skills()
                if query in name.casefold()
            ]
            self._show_completion("skills", matches)
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

    def _show_completion(
        self, mode: Literal["commands", "models", "skills"], matches: list[str]
    ) -> None:
        popup = self.query_one("#completion-popup", Vertical)
        title = self.query_one("#completion-title", Label)
        choices = self.query_one("#completion-options", OptionList)
        self._completion_mode = mode
        self._completion_matches = matches
        title.update(
            "Commands"
            if mode == "commands"
            else "Choose model"
            if mode == "models"
            else "Choose skill"
        )
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
            if selected in {"/model", "/skill"}:
                composer.value = f"{selected} "
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
        elif self._completion_mode == "skills":
            composer.value = f"/skill {selected} "
            self._hide_completion()
        composer.focus()

    async def _run_turn(self, prompt: str, *, skill_name: str | None = None) -> None:
        try:
            async for event in self.application.run(prompt, skill_name=skill_name):
                if isinstance(event, AgentStarted):
                    self._refresh_session_metadata()
                    if event.skill_name is not None:
                        await self._mount(
                            Static(
                                f"Skill active · {event.skill_name}",
                                markup=False,
                                classes="message skill-boundary",
                            )
                        )
                elif isinstance(event, ApprovalRequested):
                    self.push_screen(ApprovalScreen(event), self._approval_selected)
                elif isinstance(event, TextDelta):
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
        details = _tool_arguments(event.arguments)
        body = Static(details, markup=False, classes="tool-body")
        panel = Collapsible(
            body,
            title=f"● {_tool_title(event)} · running",
            collapsed=True,
            classes="tool-card running",
        )
        self._tools[event.call_id] = (panel, body, _tool_title(event), details)
        await self._mount(panel)

    def _tool_finished(self, event: ToolFinished) -> None:
        item = self._tools.get(event.call_id)
        if item is None:
            return
        panel, body, title, details = item
        panel.title = f"{'✓' if not event.is_error else '✕'} {title} · {event.status}"
        panel.remove_class("running")
        panel.add_class("failed" if event.is_error else "succeeded")
        body.update(_tool_body(details, event.content))

    async def _command(self, prompt: str) -> None:
        if prompt == "/help":
            await self._notice(format_slash_help(), "info")
        elif prompt == "/mode":
            current = self.application.approval_mode()
            message = (
                f"Approval mode: {current}" if current is not None else "Approval mode: unavailable"
            )
            await self._notice(message, "info")
        elif prompt.startswith("/mode "):
            target = prompt.removeprefix("/mode ").strip()
            if target not in _MODE_CYCLE:
                await self._notice("Usage: /mode auto|ask|deny", "error")
                return
            if not self.application.set_approval_mode(cast(ApprovalMode, target)):
                await self._notice("Approval mode cannot be changed for this session.", "error")
                return
            self._update_mode_indicator()
            await self._notice(f"Approval mode switched to {target}.", "info")
        elif prompt == "/skills":
            await self._notice(
                format_skill_list(
                    self.application.available_skills(), self.application.skill_warnings()
                ),
                "info",
            )
        elif prompt == "/skill":
            await self._notice(
                "Usage: /skill <name> <task> | /skill install <source> | /skill uninstall <name>",
                "warning",
            )
        elif prompt.startswith("/skill install "):
            source = prompt.removeprefix("/skill install ").strip()
            if not source:
                await self._notice("Usage: /skill install <owner/repo or url>", "warning")
                return
            await self._install_skill(source)
        elif prompt.startswith("/skill uninstall "):
            name = prompt.removeprefix("/skill uninstall ").strip()
            if not name:
                await self._notice("Usage: /skill uninstall <name>", "warning")
                return
            await self._uninstall_skill(name)
        elif prompt.startswith("/skill "):
            parts = prompt.split(maxsplit=2)
            if len(parts) < 3 or not parts[2].strip():
                await self._notice("Usage: /skill <name> <task>", "warning")
                return
            skill_name, task = parts[1], parts[2].strip()
            available = {
                name for name, _description, _source in self.application.available_skills()
            }
            if skill_name not in available:
                await self._notice(f"Unknown skill: {skill_name}. Use /skills.", "error")
                return
            await self._start_turn(task, skill_name=skill_name, history_prompt=prompt)
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
            await self._mount_welcome()
            await self._notice("Started a new empty session.", "info")
        elif prompt == "/resume":
            if self._turn_worker is not None and self._turn_worker.is_running:
                await self._notice("'/resume' is disabled while a task is in progress.", "warning")
                return
            if self.list_sessions is None or self.resume_session is None:
                await self._notice("Session selection is unavailable.", "error")
                return
            try:
                sessions = await self.list_sessions()
            except Exception:
                await self._notice("Unable to list saved sessions.", "error")
                return
            self.push_screen(
                ResumeSessionScreen(sessions, current_session_id=self.session_id),
                self._resume_selected,
            )
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
            progress = CompactionProgress()
            await self._mount(progress)
            composer = self.query_one("#composer", PromptTextArea)
            composer.disabled = True
            try:
                checkpoint = await self.application.compact_context()
            except Exception:
                progress.finish("Context compaction failed; original context retained.", "error")
                return
            finally:
                composer.disabled = False
                composer.focus()
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
            progress.finish(message)
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
        self.query_one("#brand", Static).update(self._welcome_text())
        self._refresh_session_metadata()
        self._update_mode_indicator()

    def _resume_selected(self, session_id: str | None) -> None:
        if session_id is None:
            return
        self.run_worker(
            self._resume_session(session_id), group="session-transition", exclusive=True
        )

    def _approval_selected(self, result: tuple[str, ApprovalDecision] | None) -> None:
        if result is None:
            return
        request_id, decision = result
        self.run_worker(
            self.application.resolve_approval(request_id, decision),
            group="approval-resolution",
            exclusive=True,
        )

    async def _resume_session(self, session_id: str) -> None:
        if self._turn_worker is not None and self._turn_worker.is_running:
            await self._notice("'/resume' is disabled while a task is in progress.", "warning")
            return
        if self.resume_session is None:
            await self._notice("Session selection is unavailable.", "error")
            return
        composer = self.query_one("#composer", PromptTextArea)
        composer.disabled = True
        try:
            transition = await self.resume_session(session_id)
        except RuntimeConfigurationError as error:
            await self._notice(f"Session resume failed: {error}", "error")
            return
        except Exception:
            await self._notice("Session resume failed.", "error")
            return
        finally:
            composer.disabled = False
            composer.focus()
        self._install_transition(transition)
        await self.query_one("#conversation", VerticalScroll).remove_children()
        await self._mount_welcome()
        self._prompt_history.clear()
        self._history_index = None
        self._last_history_text = None
        await self._render_recovered_history()

    async def _mount(self, widget: Static | Collapsible) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        await conversation.mount(widget)
        conversation.scroll_end(animate=False)

    async def _mount_welcome(self) -> None:
        await self._mount(Static(self._welcome_text(), markup=False, id="brand"))

    async def _notice(self, message: str, kind: str) -> None:
        await self._mount(Static(message, markup=False, classes=f"message notice {kind}"))

    def _context_label(self) -> str:
        status = self.application.context_status()
        if status is None:
            return "context unavailable"
        compacted = " · compacted" if self.application.context_checkpoint() is not None else ""
        return f"context ~{status.used_tokens}/{status.context_window} · {status.level}{compacted}"

    def _refresh_context_label(self) -> None:
        self.query_one("#status-context", Label).update(self._context_label())

    def _refresh_session_metadata(self) -> None:
        visible = bool(self.application.conversation_history())
        self.query_one("#status-context", Label).display = visible
        self.query_one("#status-session", Label).display = visible

    def _welcome_text(self) -> str:
        return (
            "   _________            codingAgent v"
            f"{self.version}\n"
            "  / ____/   |           model        "
            f"{self.model}\n"
            " / /   / /| |           workspace    "
            f"{self.workspace}\n"
            "/ /___/ ___ |           permissions  "
            f"{self.permissions}\n"
            "\\____/_/  |_|\n\n"
            "Tip: Type /help for commands or /resume to continue previous work."
        )

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

    async def _render_recovered_history(self) -> None:
        history = self.application.conversation_history()
        if not history:
            return
        compactions_by_index: dict[int, list[CompactionRecord]] = {}
        for record in self.application.compaction_history():
            compactions_by_index.setdefault(record.exchange_index, []).append(record)
        self._prompt_history.extend(
            exchange.content for exchange in history if isinstance(exchange, UserExchange)
        )
        for index, exchange in enumerate(history):
            for record in compactions_by_index.get(index, ()):
                await self._mount_compaction_record(record)
            await self._render_recovered_exchange(exchange)
        for record in compactions_by_index.get(len(history), ()):
            await self._mount_compaction_record(record)

    async def _mount_compaction_record(self, record: CompactionRecord) -> None:
        payload = record.payload
        strategy = payload.get("strategy", "legacy")
        before = payload.get("before_tokens", "?")
        after = payload.get("after_tokens", "?")
        replaced = payload.get("retained_from", "?")
        await self._mount(
            Static(
                f"Context compacted here · {strategy} summary · {before} → {after} tokens · "
                f"replaced first {replaced} exchanges. Original messages remain visible.",
                markup=False,
                classes="message context-boundary",
            )
        )

    async def _render_recovered_exchange(self, exchange: ConversationExchange) -> None:
        if isinstance(exchange, UserExchange):
            await self._mount(
                Static(exchange.content, markup=False, classes="message user-message")
            )
            return
        if isinstance(exchange, ToolContinuationExchange):
            await self._render_recovered_assistant(exchange.assistant, exchange.results)
            return
        await self._render_recovered_assistant(exchange, ())

    async def _render_recovered_assistant(
        self,
        exchange: AssistantExchange,
        results: tuple[ToolResultBlock, ...],
    ) -> None:
        result_by_id = {result.tool_use_id: result for result in results}
        for block in exchange.blocks:
            if isinstance(block, TextBlock) and block.text:
                await self._mount(
                    Static(block.text, markup=False, classes="message assistant-message")
                )
            elif isinstance(block, ThinkingBlock):
                body = Static(block.thinking or "(empty)", markup=False, classes="thinking-body")
                await self._mount(
                    Collapsible(
                        body,
                        title="Thinking · recovered",
                        collapsed=not self.thinking_visible,
                        classes="thinking-card",
                    )
                )
            elif isinstance(block, RedactedThinkingBlock):
                body = Static("Provider-redacted thinking", markup=False, classes="thinking-body")
                await self._mount(
                    Collapsible(
                        body,
                        title="Thinking · redacted",
                        collapsed=True,
                        classes="thinking-card",
                    )
                )
            elif isinstance(block, ToolUseBlock):
                await self._render_recovered_tool(block, result_by_id.get(block.call_id))
            elif isinstance(block, UnknownProviderBlock):
                await self._mount(
                    Static(
                        f"Recovered unsupported provider block: {block.block_type}",
                        markup=False,
                        classes="message notice warning",
                    )
                )

    async def _render_recovered_tool(
        self, call: ToolUseBlock, result: ToolResultBlock | None
    ) -> None:
        started = ToolStarted(call.call_id, call.name, call.input)
        title = _tool_title(started)
        details = _tool_arguments(call.input)
        content = "interrupted before result" if result is None else result.content
        body = Static(_tool_body(details, content), markup=False, classes="tool-body")
        failed = result is None or result.is_error
        status = "interrupted" if result is None else ("error" if result.is_error else "done")
        panel = Collapsible(
            body,
            title=f"{'✕' if failed else '✓'} {title} · {status}",
            collapsed=True,
            classes="tool-card failed" if failed else "tool-card succeeded",
        )
        await self._mount(panel)

    async def action_toggle_thinking(self) -> None:
        self.thinking_visible = not self.thinking_visible
        if self._thinking is not None:
            self._thinking[0].collapsed = not self.thinking_visible
        state = "shown" if self.thinking_visible else "hidden"
        await self._notice(f"Thinking details: {state}.", "info")

    async def action_cycle_approval_mode(self) -> None:
        current = self.application.approval_mode()
        if current is None:
            await self._notice("Approval mode cannot be changed for this session.", "warning")
            return
        next_mode = _MODE_CYCLE[(_MODE_CYCLE.index(current) + 1) % len(_MODE_CYCLE)]
        self.application.set_approval_mode(next_mode)
        self._update_mode_indicator()
        await self._notice(f"Approval mode switched to {next_mode}.", "info")

    def _mode_label(self) -> str:
        mode = self.application.approval_mode()
        return f"mode: {mode}" if mode else "mode: unavailable"

    def _update_mode_indicator(self) -> None:
        self.query_one("#status-mode", Label).update(self._mode_label())

    def _make_installer(self) -> SkillInstaller:
        from platformdirs import user_data_path

        return SkillInstaller(
            user_dir=user_data_path("codingAgent") / "skills",
            project_dir=Path(self.workspace) / ".agents" / "skills",
        )

    async def _install_skill(self, source: str) -> None:
        progress_label = f"Installing skills from {source}..."
        progress = Static(progress_label, markup=False, classes="message notice info")
        await self._mount(progress)
        installer = self._make_installer()
        result: InstallResult | None = None
        async for item in installer.install(source):
            if isinstance(item, tuple):
                progress.update(f"{progress_label}\n{item[0]}")
            else:
                result = item
        if result is None:
            result = InstallResult((), "Install produced no result.")
        progress.remove()
        if result.installed:
            new_skills = installer.reload()
            self.application.reload_skills(new_skills)
            kind = "info"
        else:
            kind = "warning"
        await self._notice(result.message, kind)

    async def _uninstall_skill(self, name: str) -> None:
        installer = self._make_installer()
        removed = await installer.uninstall(name)
        if removed:
            new_skills = installer.reload()
            self.application.reload_skills(new_skills)
            await self._notice(f"Uninstalled skill: {name}.", "info")
        else:
            await self._notice(f"Skill not found: {name}.", "warning")

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
    if isinstance(path, str):
        if event.tool_name == "edit_file":
            start = event.arguments.get("start_line")
            end = event.arguments.get("end_line")
            return f"edit_file · {path}:{start}-{end}"
        if event.tool_name == "read_file":
            start = event.arguments.get("start_line", 1)
            end = event.arguments.get("end_line")
            suffix = f":{start}-{end}" if end is not None else f":{start}-end"
            return f"read_file · {path}{suffix}"
        return f"{event.tool_name} · {path}"
    if event.tool_name == "code_search":
        query = event.arguments.get("query")
        if isinstance(query, str):
            return f"code_search · {_preview_text(query, 80)}"
    return event.tool_name


def _tool_arguments(arguments: dict[str, object]) -> str:
    return "Input:\n" + json.dumps(_preview_value(arguments), ensure_ascii=False, indent=2)


def _tool_body(arguments: str, result: str) -> str:
    return f"{arguments}\n\nResult:\n{_preview_text(result or '(no output)', 12_000)}"


def _preview_value(value: object) -> object:
    if isinstance(value, str):
        return _preview_text(value, 4_000)
    if isinstance(value, dict):
        return {str(key): _preview_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_preview_value(child) for child in value[:100]]
    return value


def _preview_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"{value[:limit]}\n… [{omitted} characters omitted from display]"


def _display_time(value: str) -> str:
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


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
