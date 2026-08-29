from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel
from textual.widgets import Collapsible, Label, OptionList

import coding_agent.tui as tui_module
from coding_agent.application import AgentApplication
from coding_agent.approval import ConfigurableApprovalPolicy
from coding_agent.context import ContextBudget, ContextManager, TokenEstimator
from coding_agent.domain import (
    AssistantExchange,
    CompactionRecord,
    ConversationExchange,
    TextBlock,
    ThinkingBlock,
    ToolContinuationExchange,
    ToolResultBlock,
    ToolUseBlock,
    UserExchange,
)
from coding_agent.providers.base import ProviderEvent, ProviderResponseFinished
from coding_agent.providers.fake import FakeProvider
from coding_agent.sessions.jsonl import SessionSummary
from coding_agent.sessions.memory import InMemorySessionStore
from coding_agent.tools.base import ToolOutput, ToolSpec
from coding_agent.tools.catalog import ToolCatalog
from coding_agent.tools.dispatcher import ToolDispatcher
from coding_agent.tui import (
    ApprovalScreen,
    CliTransition,
    CodingAgentTui,
    CompactionProgress,
    PromptTextArea,
    ResumeSessionScreen,
    format_slash_help,
)

pytestmark = pytest.mark.asyncio


async def test_slash_help_uses_one_aligned_command_per_line() -> None:
    assert format_slash_help().splitlines() == [
        "Commands:",
        "  /help                    Show available commands",
        "  /model [provider/model]  Show or choose a model",
        "  /mode [auto|ask|deny]    Show or set approval mode",
        "  /context                 Show context usage",
        "  /compact                 Compact conversation context",
        "  /thinking                Toggle thinking details",
        "  /resume                  Resume a saved session",
        "  /clear                   Start a new empty session",
        "  /exit                    Exit codingAgent",
    ]


def application_with_response(text: str = "Finished successfully.") -> AgentApplication:
    return AgentApplication(
        FakeProvider([AssistantExchange((TextBlock(text),), "end_turn")]),
        ToolDispatcher(ToolCatalog({})),
        InMemorySessionStore(),
    )


async def test_status_bar_shows_the_live_approval_mode_left_of_the_model_name() -> None:
    application = AgentApplication(
        FakeProvider([]),
        ToolDispatcher(ToolCatalog({})),
        InMemorySessionStore(),
        approval_policy=ConfigurableApprovalPolicy("ask", frozenset({"shell"})),
    )
    app = CodingAgentTui(
        application,
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        status_bar = app.query_one("#status-bar")
        assert str(app.query_one("#status-mode", Label).render()) == "mode: ask"
        assert list(status_bar.children).index(app.query_one("#status-mode")) < list(
            status_bar.children
        ).index(app.query_one("#status-model"))


async def test_shift_tab_cycles_approval_mode_and_updates_the_indicator() -> None:
    application = AgentApplication(
        FakeProvider([]),
        ToolDispatcher(ToolCatalog({})),
        InMemorySessionStore(),
        approval_policy=ConfigurableApprovalPolicy("auto", frozenset({"shell"})),
    )
    app = CodingAgentTui(
        application,
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        mode_label = app.query_one("#status-mode", Label)
        assert str(mode_label.render()) == "mode: auto"

        await pilot.press("shift+tab")
        await pilot.pause()
        assert str(mode_label.render()) == "mode: ask"
        assert application.approval_mode() == "ask"
        assert "Approval mode switched to ask" in str(app.query_one(".notice").render())

        await pilot.press("shift+tab")
        await pilot.pause()
        assert str(mode_label.render()) == "mode: deny"

        await pilot.press("shift+tab")
        await pilot.pause()
        assert str(mode_label.render()) == "mode: auto"


async def test_mode_slash_command_shows_and_sets_approval_mode() -> None:
    application = AgentApplication(
        FakeProvider([]),
        ToolDispatcher(ToolCatalog({})),
        InMemorySessionStore(),
        approval_policy=ConfigurableApprovalPolicy("auto", frozenset({"shell"})),
    )
    app = CodingAgentTui(
        application,
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", PromptTextArea)
        composer.value = "/mode"
        await pilot.pause()
        await pilot.press("escape")
        await pilot.press("enter")
        await pilot.pause()
        assert "Approval mode: auto" in str(app.query_one(".notice").render())

        composer.value = "/mode deny"
        await pilot.pause()
        await pilot.press("escape")
        await pilot.press("enter")
        await pilot.pause()
        assert application.approval_mode() == "deny"
        assert str(app.query_one("#status-mode", Label).render()) == "mode: deny"


async def test_tui_composes_full_screen_workspace() -> None:
    app = CodingAgentTui(
        application_with_response(),
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
        available_models=("provider/model",),
        version="1.2.3",
        permissions="ask",
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#conversation")
        assert app.query_one("#composer")
        conversation = app.query_one("#conversation")
        brand = app.query_one("#brand")
        assert brand.parent is conversation
        assert conversation.children[0] is brand
        welcome = str(brand.render())
        assert "codingAgent v1.2.3" in welcome
        assert "model        provider/model" in welcome
        assert "workspace    /tmp/project" in welcome
        assert "permissions  ask" in welcome
        assert "provider/model" in str(app.query_one("#status-model").render())
        assert "/tmp/project" in str(app.query_one("#status-workspace").render())
        assert app.query_one("#status-context").display is False
        assert app.query_one("#status-session").display is False


async def test_tui_submits_prompt_and_streams_assistant_card() -> None:
    app = CodingAgentTui(
        application_with_response(),
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
        available_models=("provider/model",),
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        composer.value = "Fix the tests"
        await pilot.press("enter")
        await pilot.pause()

        assert "Fix the tests" in str(app.query_one(".user-message").render())
        assert "Finished successfully." in str(app.query_one(".assistant-message").render())
        assert composer.value == ""
        assert composer.disabled is False
        assert app.query_one("#status-context").display is True
        assert app.query_one("#status-session").display is True


async def test_composer_soft_wraps_long_input_instead_of_scrolling_horizontally() -> None:
    app = CodingAgentTui(
        application_with_response(),
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
    )

    async with app.run_test(size=(60, 24)) as pilot:
        composer = app.query_one("#composer", PromptTextArea)
        composer.value = "这是一段很长的用户输入" * 20
        await pilot.pause()

        assert composer.soft_wrap is True
        assert composer.wrapped_document.height > 1
        assert composer.scroll_x == 0
        assert composer.size.height > 3


async def test_inactive_completion_shortcuts_are_safe() -> None:
    app = CodingAgentTui(
        application_with_response(),
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
    )

    async with app.run_test():
        composer = app.query_one("#composer", PromptTextArea)
        assert composer.completion_active is False

        composer.action_completion_dismiss()
        composer.action_completion_accept()

        assert composer.completion_active is False


async def test_compaction_progress_can_finish_before_its_timer_starts() -> None:
    progress = CompactionProgress()

    progress.finish("Compaction skipped", "warning")

    assert progress.has_class("warning")
    assert "Compaction skipped" in str(progress.render())


async def test_shift_enter_adds_newline_and_enter_submits_multiline_prompt() -> None:
    app = CodingAgentTui(
        application_with_response(),
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", PromptTextArea)
        composer.value = "First line"
        await pilot.press("end", "shift+enter")
        await pilot.press(*"Second line")
        assert composer.value == "First line\nSecond line"

        await pilot.press("enter")
        await pilot.pause()

        assert "First line\nSecond line" in str(app.query_one(".user-message").render())


async def test_empty_composer_navigates_prompt_history_and_text_arrows_move_to_edges() -> None:
    provider = FakeProvider(
        [
            AssistantExchange((TextBlock("First response"),), "end_turn"),
            AssistantExchange((TextBlock("Second response"),), "end_turn"),
        ]
    )
    app = CodingAgentTui(
        AgentApplication(
            provider,
            ToolDispatcher(ToolCatalog({})),
            InMemorySessionStore(),
        ),
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", PromptTextArea)
        for prompt in ("First prompt", "Second prompt"):
            composer.value = prompt
            await pilot.press("enter")
            await pilot.pause()

        await pilot.press("up")
        await pilot.pause()
        assert composer.value == "Second prompt"

        await pilot.press("up")
        await pilot.pause()
        assert composer.value == "First prompt"

        await pilot.press("down")
        await pilot.pause()
        assert composer.value == "Second prompt"

        await pilot.press("down")
        await pilot.pause()
        assert composer.value == ""

        await pilot.press("up")
        await pilot.pause()
        assert composer.value == "Second prompt"

        await pilot.press(*" edited")
        await pilot.pause()
        assert composer.value == "Second prompt edited"

        await pilot.press("up")
        await pilot.pause()
        assert composer.cursor_location == (0, 0)
        assert composer.value == "Second prompt edited"
        await pilot.press("down")
        await pilot.pause()
        assert composer.cursor_location == (0, len("Second prompt edited"))

        composer.value = "First line\nSecond line"
        await pilot.press("up")
        await pilot.pause()
        assert composer.cursor_location == (0, 0)
        await pilot.press("down")
        await pilot.pause()
        assert composer.cursor_location == (1, len("Second line"))


async def test_slash_popup_filters_navigates_completes_and_dismisses() -> None:
    app = CodingAgentTui(
        application_with_response(),
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
        available_models=("provider/model", "provider/other"),
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", PromptTextArea)
        popup = app.query_one("#completion-popup")
        choices = app.query_one("#completion-options", OptionList)

        composer.value = "/"
        await pilot.pause()
        await pilot.pause()
        assert popup.display is True
        assert choices.option_count >= 7

        composer.value = "/model"
        await pilot.pause()
        await pilot.pause()
        assert choices.option_count == 1
        assert choices.get_option_at_index(0).id == "/model"
        assert choices.size.height >= 1

        await pilot.press("tab")
        await pilot.pause()
        assert composer.value == "/model "
        assert choices.option_count == 2
        assert choices.get_option_at_index(0).id == "provider/model"

        await pilot.press("down")
        await pilot.pause()
        assert choices.highlighted == 1
        await pilot.press("up")
        await pilot.pause()
        assert choices.highlighted == 0
        await pilot.press("tab")
        await pilot.pause()
        assert composer.value == "/model provider/model"

        await pilot.press("escape")
        await pilot.pause()
        assert popup.display is False

        await pilot.press("backspace")
        await pilot.pause()
        assert popup.display is True


async def test_slash_popup_allocates_visible_rows_for_every_command_match() -> None:
    app = CodingAgentTui(
        application_with_response(),
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
    )

    async with app.run_test(size=(80, 30)) as pilot:
        composer = app.query_one("#composer", PromptTextArea)
        choices = app.query_one("#completion-options", OptionList)

        for prefix, expected in (
            ("/c", ("/context", "/compact", "/clear")),
            ("/m", ("/model", "/mode")),
            ("/r", ("/resume",)),
            ("/h", ("/help",)),
            ("/t", ("/thinking",)),
        ):
            composer.value = prefix
            await pilot.pause()
            await pilot.pause()
            assert (
                tuple(
                    choices.get_option_at_index(index).id for index in range(choices.option_count)
                )
                == expected
            )
            assert choices.size.height >= len(expected)


async def test_resume_command_opens_full_screen_picker_and_installs_selection() -> None:
    initial = application_with_response()
    resumed = AgentApplication(
        FakeProvider([]),
        ToolDispatcher(ToolCatalog({})),
        InMemorySessionStore(),
        initial_exchanges=(
            UserExchange("Selected historical task"),
            AssistantExchange((TextBlock("Historical answer"),), "end_turn"),
        ),
    )
    sessions = (
        SessionSummary(
            "session-1",
            "Current task",
            "2026-08-28T08:00:00+00:00",
            "2026-08-28T09:00:00+00:00",
            "provider/model",
            4,
            False,
        ),
        SessionSummary(
            "session-2",
            "Selected historical task",
            "2026-08-27T08:00:00+00:00",
            "2026-08-27T09:00:00+00:00",
            "provider/other",
            2,
            True,
        ),
    )
    resumed_ids: list[str] = []

    async def list_sessions() -> tuple[SessionSummary, ...]:
        return sessions

    async def resume_session(session_id: str) -> CliTransition:
        resumed_ids.append(session_id)
        return CliTransition(resumed, "provider/model", session_id, ("provider/model",))

    app = CodingAgentTui(
        initial,
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
        available_models=("provider/model",),
        list_sessions=list_sessions,
        resume_session=resume_session,
    )

    async with app.run_test() as pilot:
        await app._command("/resume")
        await pilot.pause()

        assert isinstance(app.screen, ResumeSessionScreen)
        options = app.screen.query_one("#resume-options", OptionList)
        assert options.option_count == 2
        assert "Current task" in str(options.get_option_at_index(0).prompt)
        assert "current" in str(options.get_option_at_index(0).prompt)
        assert "Selected historical task" in str(options.get_option_at_index(1).prompt)

        await pilot.press("down", "enter")
        await pilot.pause()
        await pilot.pause()

        assert resumed_ids == ["session-2"]
        assert app.session_id == "session-2"
        rendered = "\n".join(
            str(child.render()) for child in app.query_one("#conversation").children
        )
        conversation = app.query_one("#conversation")
        assert conversation.children[0].id == "brand"
        assert "Selected historical task" in rendered
        assert "Historical answer" in rendered


async def test_resume_is_explicitly_blocked_while_a_turn_is_running() -> None:
    class SlowProvider:
        async def stream(
            self,
            conversation: tuple[ConversationExchange, ...],
            tools: tuple[ToolSpec, ...],
            system_instructions: str | None = None,
        ) -> AsyncIterator[ProviderEvent]:
            del conversation, tools, system_instructions
            await asyncio.Event().wait()
            if False:
                yield

    list_called = False

    async def list_sessions() -> tuple[SessionSummary, ...]:
        nonlocal list_called
        list_called = True
        return ()

    async def resume_session(_session_id: str) -> CliTransition:
        raise AssertionError("resume callback must not run during an active turn")

    app = CodingAgentTui(
        AgentApplication(SlowProvider(), ToolDispatcher(ToolCatalog({})), InMemorySessionStore()),
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
        list_sessions=list_sessions,
        resume_session=resume_session,
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", PromptTextArea)
        composer.value = "Keep working"
        await pilot.press("enter")
        await pilot.pause()

        await app._command("/resume")
        await pilot.pause()

        assert list_called is False
        assert "disabled while a task is in progress" in str(
            list(app.query(".notice"))[-1].render()
        )
        await app.action_cancel_turn()


async def test_tui_approval_screen_shows_command_and_allows_once() -> None:
    class ShellInput(BaseModel):
        command: str

    class Shell:
        name = "shell"
        description = "Run a command"
        input_model = ShellInput

        def __init__(self) -> None:
            self.executed = False

        async def execute(self, arguments: BaseModel) -> ToolOutput:
            self.executed = True
            return ToolOutput("exit_code: 0")

    shell = Shell()
    application = AgentApplication(
        FakeProvider(
            [
                AssistantExchange(
                    (ToolUseBlock("call-1", "shell", {"command": "uv run pytest"}),),
                    "tool_use",
                ),
                AssistantExchange((TextBlock("done"),), "end_turn"),
            ]
        ),
        ToolDispatcher(ToolCatalog({"shell": shell})),
        InMemorySessionStore(),
        approval_policy=ConfigurableApprovalPolicy("ask", frozenset({"shell"})),
    )
    app = CodingAgentTui(
        application,
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
    )

    async with app.run_test() as pilot:
        app.query_one("#composer", PromptTextArea).value = "Run tests"
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, ApprovalScreen)
        assert "uv run pytest" in str(app.screen.query_one("#approval-details-text").render())
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert shell.executed is True
        assert "done" in str(app.query_one(".assistant-message").render())


async def test_slash_popup_handles_plain_completion_and_no_matches() -> None:
    app = CodingAgentTui(
        application_with_response(),
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", PromptTextArea)
        popup = app.query_one("#completion-popup")
        choices = app.query_one("#completion-options", OptionList)

        composer.value = "/con"
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert composer.value == "/context"
        await pilot.press("enter")
        await pilot.pause()
        assert "Context management is unavailable." in str(app.query_one(".notice").render())

        composer.value = "/does-not-exist"
        await pilot.pause()
        assert popup.display is True
        assert choices.highlighted is None
        assert "No matches" in str(choices.get_option_at_index(0).prompt)
        await pilot.press("enter")
        await pilot.pause()
        assert composer.value == "/does-not-exist"

        composer.value = "/context extra"
        await pilot.pause()
        assert popup.display is False


async def test_model_command_opens_secondary_picker_and_switches_selection() -> None:
    initial = application_with_response()
    switched = application_with_response()
    selected: list[str | None] = []

    async def switch_model(target: str | None) -> CliTransition:
        selected.append(target)
        return CliTransition(switched, target or "", "session-2", ("model-a", "model-b"))

    app = CodingAgentTui(
        initial,
        model="model-a",
        workspace="/tmp/project",
        session_id="session-1",
        available_models=("model-a", "model-b"),
        switch_model=switch_model,
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", PromptTextArea)
        choices = app.query_one("#completion-options", OptionList)
        composer.value = "/model"
        await pilot.pause()

        await pilot.press("enter")
        await pilot.pause()
        assert composer.value == "/model "
        assert choices.option_count == 2

        await pilot.press("down", "enter")
        await pilot.pause()

        assert selected == ["model-b"]
        assert app.model == "model-b"
        assert composer.value == ""
        assert app.query_one("#completion-popup").display is False


async def test_tui_renders_collapsible_thinking_and_completed_tool_card() -> None:
    class ShellInput(BaseModel):
        command: str
        cwd: str = "."

    class Shell:
        name = "shell"
        description = "Run a test command"
        input_model = ShellInput

        async def execute(self, arguments: BaseModel) -> ToolOutput:
            return ToolOutput("exit_code: 0\nstdout:\nall green")

    provider = FakeProvider(
        [
            AssistantExchange(
                (
                    ThinkingBlock("inspect"),
                    ToolUseBlock(
                        "call-1",
                        "shell",
                        {"command": "uv run pytest", "cwd": "tests"},
                    ),
                ),
                "tool_use",
            ),
            AssistantExchange((TextBlock("Done."),), "end_turn"),
        ]
    )
    app = CodingAgentTui(
        AgentApplication(
            provider,
            ToolDispatcher(ToolCatalog({"shell": Shell()})),
            InMemorySessionStore(),
        ),
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
    )

    async with app.run_test() as pilot:
        app.query_one("#composer").value = "Run tests"
        await pilot.press("enter")
        await pilot.pause()

        thinking = app.query_one(".thinking-card", Collapsible)
        tool = app.query_one(".tool-card", Collapsible)
        assert thinking.collapsed is True
        assert thinking.title == "Thinking · complete"
        assert "uv run pytest" in tool.title
        assert "[tests]" in tool.title
        assert "done" in tool.title
        assert '"command": "uv run pytest"' in str(tool.query_one(".tool-body").render())
        assert "Result:" in str(tool.query_one(".tool-body").render())
        assert "all green" in str(tool.query_one(".tool-body").render())


async def test_tui_replays_full_history_with_compaction_boundary_and_tool_inputs() -> None:
    tool_call = ToolUseBlock(
        "write-1",
        "write_file",
        {"path": "hello.py", "content": "print('visible after resume')\n"},
    )
    history: tuple[ConversationExchange, ...] = (
        UserExchange("old request that the model no longer receives verbatim"),
        ToolContinuationExchange(
            AssistantExchange((TextBlock("Creating the file."), tool_call), "tool_use"),
            (ToolResultBlock("write-1", "Created hello.py (30 bytes)", False),),
        ),
        UserExchange("recent request"),
        AssistantExchange((TextBlock("recent answer"),), "end_turn"),
    )
    summary = (
        "task_goal: create hello.py\n"
        "user_constraints: preserve visible history\n"
        "decisions: wrote the file\n"
        "files_read: none\n"
        "files_modified: hello.py\n"
        "commands_and_results: write succeeded\n"
        "verification_status: pending\n"
        "known_failures: none\n"
        "pending_work: continue"
    )
    manager = ContextManager(
        ContextBudget(context_window=20_000, max_output_tokens=2_000),
        TokenEstimator(),
        retained_exchanges=2,
    )
    manager.restore(
        history,
        {
            "reason": "manual",
            "strategy": "provider",
            "retained_from": 2,
            "before_tokens": 1_000,
            "after_tokens": 500,
            "summary": summary,
        },
    )
    application = AgentApplication(
        FakeProvider([]),
        ToolDispatcher(ToolCatalog({})),
        InMemorySessionStore(),
        initial_exchanges=history,
        context_manager=manager,
        initial_compactions=(
            CompactionRecord(
                2,
                {
                    "strategy": "provider",
                    "retained_from": 2,
                    "before_tokens": 1_000,
                    "after_tokens": 500,
                },
            ),
        ),
    )
    app = CodingAgentTui(
        application,
        model="provider/model",
        workspace="/tmp/project",
        session_id="resumed-session",
    )

    async with app.run_test() as pilot:
        await pilot.pause()

        conversation = app.query_one("#conversation")
        rendered = "\n".join(str(child.render()) for child in conversation.children)
        assert "old request that the model no longer receives verbatim" in rendered
        assert "recent request" in rendered
        assert "recent answer" in rendered
        assert "Context compacted here" in rendered
        assert "compacted" in str(app.query_one("#status-context").render())
        tool_body = str(app.query_one(".tool-body").render())
        assert "print('visible after resume')" in tool_body
        assert "Created hello.py (30 bytes)" in tool_body

        composer = app.query_one("#composer", PromptTextArea)
        await pilot.press("up")
        assert composer.value == "recent request"
        await pilot.press("up")
        assert composer.value == "old request that the model no longer receives verbatim"


async def test_manual_compaction_shows_indeterminate_progress_until_provider_finishes() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    summary = (
        "task_goal: continue\n"
        "user_constraints: none\n"
        "decisions: none\n"
        "files_read: none\n"
        "files_modified: none\n"
        "commands_and_results: none\n"
        "verification_status: pending\n"
        "known_failures: none\n"
        "pending_work: continue"
    )

    class BlockingSummaryProvider:
        async def stream(
            self,
            conversation: tuple[ConversationExchange, ...],
            tools: tuple[ToolSpec, ...],
            system_instructions: str | None = None,
        ) -> AsyncIterator[ProviderEvent]:
            del conversation, tools, system_instructions
            started.set()
            await release.wait()
            yield ProviderResponseFinished(AssistantExchange((TextBlock(summary),), "end_turn"))

    history: tuple[ConversationExchange, ...] = tuple(
        exchange
        for index in range(4)
        for exchange in (
            UserExchange(f"question {index} " * 30),
            AssistantExchange((TextBlock(f"answer {index} " * 30),), "end_turn"),
        )
    )
    application = AgentApplication(
        BlockingSummaryProvider(),
        ToolDispatcher(ToolCatalog({})),
        InMemorySessionStore(),
        initial_exchanges=history,
        context_manager=ContextManager(
            ContextBudget(context_window=20_000, max_output_tokens=2_000),
            TokenEstimator(),
            retained_exchanges=2,
        ),
    )
    app = CodingAgentTui(
        application,
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
    )

    async with app.run_test() as pilot:
        task = asyncio.create_task(app._command("/compact"))
        await started.wait()
        await pilot.pause()

        progress = app.query_one(".compaction-progress", CompactionProgress)
        assert "Compacting context" in str(progress.render())
        assert app.query_one("#composer", PromptTextArea).disabled is True

        release.set()
        await task
        await pilot.pause()
        assert "Compacted context with provider summary" in str(progress.render())
        assert app.query_one("#composer", PromptTextArea).disabled is False


async def test_tui_preserves_text_and_tool_chronology() -> None:
    class ReadInput(BaseModel):
        path: str

    class ReadFile:
        name = "read_file"
        description = "Read a file"
        input_model = ReadInput

        async def execute(self, arguments: BaseModel) -> ToolOutput:
            return ToolOutput("task contents")

    provider = FakeProvider(
        [
            AssistantExchange(
                (
                    TextBlock("I will inspect the task."),
                    ToolUseBlock("call-1", "read_file", {"path": "TASK.md"}),
                ),
                "tool_use",
            ),
            AssistantExchange((TextBlock("The task is complete."),), "end_turn"),
        ]
    )
    app = CodingAgentTui(
        AgentApplication(
            provider,
            ToolDispatcher(ToolCatalog({"read_file": ReadFile()})),
            InMemorySessionStore(),
        ),
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
    )

    async with app.run_test() as pilot:
        app.query_one("#composer").value = "Do the task"
        await pilot.press("enter")
        await pilot.pause()

        conversation = app.query_one("#conversation")
        timeline = [
            "user"
            if child.has_class("user-message")
            else "assistant"
            if child.has_class("assistant-message")
            else "tool"
            if child.has_class("tool-card")
            else "other"
            for child in conversation.children
        ]
        assert timeline == ["other", "user", "assistant", "tool", "assistant"]
        replies = app.query(".assistant-message")
        assert "I will inspect the task." in str(replies[0].render())
        assert "The task is complete." in str(replies[1].render())


async def test_tui_slash_commands_update_state_without_leaving_full_screen() -> None:
    initial = application_with_response()
    switched = application_with_response()
    cleared = application_with_response()

    async def switch_model(target: str | None) -> CliTransition:
        assert target == "provider/other"
        return CliTransition(switched, target, "session-1", ("provider/model", target))

    async def clear_session(_target: str | None) -> CliTransition:
        return CliTransition(cleared, "provider/other", "session-2", ("provider/other",))

    app = CodingAgentTui(
        initial,
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
        available_models=("provider/model", "provider/other"),
        switch_model=switch_model,
        clear_session=clear_session,
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        for command in (
            "/help",
            "/model",
            "/thinking",
            "/thinking",
            "/context",
            "/compact",
            "/unknown",
            "/model provider/other",
            "/clear",
        ):
            composer.value = command
            await pilot.press("enter")
            await pilot.pause()

        assert app.model == "provider/other"
        assert app.session_id == "session-2"
        assert "provider/other" in str(app.query_one("#status-model").render())
        assert "session session-" in str(app.query_one("#status-session").render())
        assert app.query_one("#conversation").children[0].id == "brand"
        assert "Started a new empty session." in str(app.query_one(".notice").render())


async def test_manual_compaction_refreshes_context_and_reports_projection_details() -> None:
    history: list[ConversationExchange] = []
    for index in range(4):
        history.extend(
            (
                UserExchange(f"old question {index} " * 60),
                AssistantExchange((TextBlock(f"old answer {index} " * 60),), "end_turn"),
            )
        )
    summary = (
        "task_goal: continue the test\n"
        "user_constraints: keep output short\n"
        "decisions: none\n"
        "files_read: none\n"
        "files_modified: none\n"
        "commands_and_results: none\n"
        "verification_status: pending\n"
        "known_failures: none\n"
        "pending_work: continue"
    )
    context_manager = ContextManager(
        ContextBudget(context_window=20_000, max_output_tokens=2_000),
        TokenEstimator(),
        retained_exchanges=2,
    )
    context_manager.record_provider_usage(history, {"input_tokens": 2_400})
    application = AgentApplication(
        FakeProvider([AssistantExchange((TextBlock(summary),), "end_turn")]),
        ToolDispatcher(ToolCatalog({})),
        InMemorySessionStore(),
        initial_exchanges=history,
        context_manager=context_manager,
    )
    app = CodingAgentTui(
        application,
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
    )

    async with app.run_test() as pilot:
        before = str(app.query_one("#status-context").render())
        await app._command("/compact")
        await pilot.pause()

        status = application.context_status()
        assert status is not None
        assert str(status.used_tokens) in str(app.query_one("#status-context").render())
        assert str(app.query_one("#status-context").render()) != before
        notice = str(list(app.query(".notice"))[-1].render())
        assert "provider summary" in notice
        assert "replaced 6 exchanges" in notice
        assert "retained 2" in notice

        await app._command("/context")
        await pilot.pause()
        context_notice = str(list(app.query(".notice"))[-1].render())
        assert "Context estimate:" in context_notice
        assert "last Provider input=2400 tokens exact" in context_notice
        assert "Last compaction: provider summary" in context_notice


async def test_ctrl_c_cancels_active_provider_without_closing_tui() -> None:
    class SlowProvider:
        async def stream(
            self,
            conversation: tuple[ConversationExchange, ...],
            tools: tuple[ToolSpec, ...],
            system_instructions: str | None = None,
        ) -> AsyncIterator[ProviderEvent]:
            del conversation, tools, system_instructions
            await asyncio.Event().wait()
            if False:
                yield

    store = InMemorySessionStore()
    app = CodingAgentTui(
        AgentApplication(SlowProvider(), ToolDispatcher(ToolCatalog({})), store),
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer")
        composer.value = "Wait"
        await pilot.press("enter")
        await pilot.pause()
        assert composer.disabled is True

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert composer.disabled is False
        assert "turn_cancelled" in store.kinds


async def test_ctrl_c_copies_selected_conversation_text_when_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_copies: list[str] = []

    async def copy_native(text: str) -> None:
        native_copies.append(text)

    monkeypatch.setattr(tui_module, "_copy_to_macos_clipboard", copy_native)
    app = CodingAgentTui(
        application_with_response(),
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
    )

    async with app.run_test() as pilot:
        app.query_one("#composer").value = "Fix the tests"
        await pilot.press("enter")
        await pilot.pause()

        app.query_one(".assistant-message").text_select_all()
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert app.clipboard == "Finished successfully."
        assert native_copies == ["Finished successfully."]


@pytest.mark.skipif(sys.platform != "darwin", reason="Command+C is a macOS shortcut")
async def test_command_c_copies_selected_composer_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_copies: list[str] = []

    async def copy_native(text: str) -> None:
        native_copies.append(text)

    monkeypatch.setattr(tui_module, "_copy_to_macos_clipboard", copy_native)
    app = CodingAgentTui(
        application_with_response(),
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
    )

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", PromptTextArea)
        composer.value = "copy this input"
        composer.select_all()
        await pilot.press("super+c")
        await pilot.pause()

        assert app.clipboard == "copy this input"
        assert native_copies == ["copy this input"]
