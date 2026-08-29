from __future__ import annotations

import pytest
from pydantic import BaseModel

from coding_agent.application import AgentApplication
from coding_agent.approval import ConfigurableApprovalPolicy
from coding_agent.domain import AssistantExchange, TextBlock, ToolContinuationExchange, ToolUseBlock
from coding_agent.events import ApprovalRequested, ToolFinished
from coding_agent.providers.fake import FakeProvider
from coding_agent.sessions.memory import InMemorySessionStore
from coding_agent.tools.base import ToolOutput
from coding_agent.tools.catalog import ToolCatalog
from coding_agent.tools.dispatcher import ToolDispatcher


class CommandInput(BaseModel):
    command: str


class RecordingShell:
    name = "shell"
    description = "Run a command"
    input_model = CommandInput

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def execute(self, arguments: BaseModel) -> ToolOutput:
        parsed = CommandInput.model_validate(arguments)
        self.commands.append(parsed.command)
        return ToolOutput("exit_code: 0")


@pytest.mark.asyncio
async def test_ask_policy_waits_for_approval_then_executes() -> None:
    shell = RecordingShell()
    sessions = InMemorySessionStore()
    application = AgentApplication(
        FakeProvider(
            [
                AssistantExchange(
                    (ToolUseBlock("call-1", "shell", {"command": "pytest"}),), "tool_use"
                ),
                AssistantExchange((TextBlock("done"),), "end_turn"),
            ]
        ),
        ToolDispatcher(ToolCatalog({"shell": shell})),
        sessions,
        approval_policy=ConfigurableApprovalPolicy("ask", frozenset({"shell"})),
    )
    events = []

    async for event in application.run("test"):
        events.append(event)
        if isinstance(event, ApprovalRequested):
            assert shell.commands == []
            assert await application.resolve_approval(event.request_id, "allow_once") is True

    assert shell.commands == ["pytest"]
    assert sum(isinstance(event, ApprovalRequested) for event in events) == 1
    assert "approval_requested" in sessions.kinds
    assert "approval_resolved" in sessions.kinds


@pytest.mark.asyncio
async def test_denied_tool_is_not_executed_and_is_returned_to_model() -> None:
    shell = RecordingShell()
    provider = FakeProvider(
        [
            AssistantExchange(
                (ToolUseBlock("call-1", "shell", {"command": "rm file"}),), "tool_use"
            ),
            AssistantExchange((TextBlock("permission was denied"),), "end_turn"),
        ]
    )
    sessions = InMemorySessionStore()
    application = AgentApplication(
        provider,
        ToolDispatcher(ToolCatalog({"shell": shell})),
        sessions,
        approval_policy=ConfigurableApprovalPolicy("ask", frozenset({"shell"})),
    )
    events = []

    async for event in application.run("delete"):
        events.append(event)
        if isinstance(event, ApprovalRequested):
            await application.resolve_approval(event.request_id, "deny")

    assert shell.commands == []
    denied = next(event for event in events if isinstance(event, ToolFinished))
    assert denied.metadata == {"denied": True}
    continuation = next(
        record.payload for record in sessions.records if record.kind == "tool_continuation"
    )
    assert isinstance(continuation, ToolContinuationExchange)
    assert continuation.results[0].is_error is True
    assert "not executed" in continuation.results[0].content


@pytest.mark.asyncio
async def test_allow_session_skips_later_prompts_for_the_same_tool() -> None:
    shell = RecordingShell()
    application = AgentApplication(
        FakeProvider(
            [
                AssistantExchange(
                    (
                        ToolUseBlock("call-1", "shell", {"command": "one"}),
                        ToolUseBlock("call-2", "shell", {"command": "two"}),
                    ),
                    "tool_use",
                ),
                AssistantExchange((TextBlock("done"),), "end_turn"),
            ]
        ),
        ToolDispatcher(ToolCatalog({"shell": shell})),
        InMemorySessionStore(),
        approval_policy=ConfigurableApprovalPolicy("ask", frozenset({"shell"})),
    )
    requests = 0

    async for event in application.run("run both"):
        if isinstance(event, ApprovalRequested):
            requests += 1
            await application.resolve_approval(event.request_id, "allow_session")

    assert requests == 1
    assert shell.commands == ["one", "two"]


@pytest.mark.asyncio
async def test_classify_override_forces_ask_even_in_auto_mode() -> None:
    shell = RecordingShell()
    application = AgentApplication(
        FakeProvider(
            [
                AssistantExchange(
                    (ToolUseBlock("call-1", "shell", {"command": "rm -rf /"}),), "tool_use"
                ),
                AssistantExchange((TextBlock("done"),), "end_turn"),
            ]
        ),
        ToolDispatcher(ToolCatalog({"shell": shell})),
        InMemorySessionStore(),
        approval_policy=ConfigurableApprovalPolicy(
            "auto", frozenset(), classify=lambda call: "ask" if call.name == "shell" else None
        ),
    )
    events = []

    async for event in application.run("clean up"):
        events.append(event)
        if isinstance(event, ApprovalRequested):
            await application.resolve_approval(event.request_id, "allow_once")

    assert sum(isinstance(event, ApprovalRequested) for event in events) == 1
    assert shell.commands == ["rm -rf /"]


@pytest.mark.asyncio
async def test_classify_override_forces_allow_bypassing_guarded_tools() -> None:
    shell = RecordingShell()
    application = AgentApplication(
        FakeProvider(
            [
                AssistantExchange(
                    (ToolUseBlock("call-1", "shell", {"command": "git status"}),), "tool_use"
                ),
                AssistantExchange((TextBlock("done"),), "end_turn"),
            ]
        ),
        ToolDispatcher(ToolCatalog({"shell": shell})),
        InMemorySessionStore(),
        approval_policy=ConfigurableApprovalPolicy(
            "ask", frozenset({"shell"}), classify=lambda call: "allow"
        ),
    )

    events = [event async for event in application.run("check status")]

    assert not any(isinstance(event, ApprovalRequested) for event in events)
    assert shell.commands == ["git status"]


@pytest.mark.asyncio
async def test_classify_override_ignores_allow_session_for_forced_ask() -> None:
    shell = RecordingShell()
    application = AgentApplication(
        FakeProvider(
            [
                AssistantExchange(
                    (
                        ToolUseBlock("call-1", "shell", {"command": "pwd"}),
                        ToolUseBlock("call-2", "shell", {"command": "rm -rf /"}),
                    ),
                    "tool_use",
                ),
                AssistantExchange((TextBlock("done"),), "end_turn"),
            ]
        ),
        ToolDispatcher(ToolCatalog({"shell": shell})),
        InMemorySessionStore(),
        approval_policy=ConfigurableApprovalPolicy(
            "ask",
            frozenset({"shell"}),
            classify=lambda call: "ask" if "rm" in call.input.get("command", "") else None,
        ),
    )
    requests = 0

    async for event in application.run("run both"):
        if isinstance(event, ApprovalRequested):
            requests += 1
            await application.resolve_approval(event.request_id, "allow_session")

    assert requests == 2
    assert shell.commands == ["pwd", "rm -rf /"]


@pytest.mark.asyncio
async def test_deny_policy_never_requests_or_executes() -> None:
    shell = RecordingShell()
    application = AgentApplication(
        FakeProvider(
            [
                AssistantExchange(
                    (ToolUseBlock("call-1", "shell", {"command": "anything"}),), "tool_use"
                ),
                AssistantExchange((TextBlock("done"),), "end_turn"),
            ]
        ),
        ToolDispatcher(ToolCatalog({"shell": shell})),
        InMemorySessionStore(),
        approval_policy=ConfigurableApprovalPolicy("deny", frozenset({"shell"})),
    )

    events = [event async for event in application.run("run")]

    assert shell.commands == []
    assert not any(isinstance(event, ApprovalRequested) for event in events)
