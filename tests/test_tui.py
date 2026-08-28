from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel
from textual.widgets import Collapsible

from coding_agent.application import AgentApplication
from coding_agent.domain import (
    AssistantExchange,
    ConversationExchange,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from coding_agent.providers.base import ProviderEvent
from coding_agent.providers.fake import FakeProvider
from coding_agent.sessions.memory import InMemorySessionStore
from coding_agent.tools.base import ToolOutput, ToolSpec
from coding_agent.tools.catalog import ToolCatalog
from coding_agent.tools.dispatcher import ToolDispatcher
from coding_agent.tui import CliTransition, CodingAgentTui

pytestmark = pytest.mark.asyncio


def application_with_response(text: str = "Finished successfully.") -> AgentApplication:
    return AgentApplication(
        FakeProvider([AssistantExchange((TextBlock(text),), "end_turn")]),
        ToolDispatcher(ToolCatalog({})),
        InMemorySessionStore(),
    )


async def test_tui_composes_full_screen_workspace() -> None:
    app = CodingAgentTui(
        application_with_response(),
        model="provider/model",
        workspace="/tmp/project",
        session_id="session-1",
        available_models=("provider/model",),
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#conversation")
        assert app.query_one("#composer")
        assert "provider/model" in str(app.query_one("#status-model").render())
        assert "/tmp/project" in str(app.query_one("#status-workspace").render())


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
        assert "all green" in str(tool.query_one(".tool-body").render())


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
        assert "Started a new empty session." in str(app.query_one(".notice").render())


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
