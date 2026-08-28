from __future__ import annotations

import asyncio
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from coding_agent.application import AgentApplication
from coding_agent.domain import (
    AssistantExchange,
    ConversationExchange,
    TextBlock,
    ToolContinuationExchange,
    ToolUseBlock,
)
from coding_agent.events import (
    AgentCancelled,
    AgentCompleted,
    AgentFailed,
    TextDelta,
    ToolFinished,
    ToolStarted,
    WarningRaised,
)
from coding_agent.providers.base import ProviderEvent, ProviderTextDelta
from coding_agent.providers.fake import FakeProvider
from coding_agent.sessions.jsonl import Redactor
from coding_agent.sessions.memory import InMemorySessionStore
from coding_agent.tools.base import ToolOutput, ToolSpec
from coding_agent.tools.builtin import BuiltinToolSource
from coding_agent.tools.catalog import ToolCatalog
from coding_agent.tools.dispatcher import ToolDispatcher


@pytest.fixture
def workspace() -> Path:
    path = Path.cwd() / ".tmp" / "walking-skeleton"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.asyncio
async def test_user_request_can_read_a_file_and_finish(workspace: Path) -> None:
    (workspace / "hello.txt").write_text("hello from the workspace\n", encoding="utf-8")
    provider = FakeProvider(
        responses=[
            AssistantExchange(
                blocks=(
                    ToolUseBlock(
                        call_id="call-1",
                        name="read_file",
                        input={"path": "hello.txt"},
                    ),
                ),
                stop_reason="tool_use",
            ),
            AssistantExchange(
                blocks=(TextBlock(text="The file says hello from the workspace."),),
                stop_reason="end_turn",
            ),
        ]
    )
    sessions = InMemorySessionStore()
    catalog = await ToolCatalog.create([BuiltinToolSource(workspace)])
    application = AgentApplication(
        provider=provider,
        dispatcher=ToolDispatcher(catalog),
        sessions=sessions,
        max_steps=4,
    )

    events = [event async for event in application.run("Read hello.txt")]

    assert TextDelta(text="The file says hello from the workspace.") in events
    assert (
        ToolFinished(
            call_id="call-1",
            tool_name="read_file",
            is_error=False,
            content="1: hello from the workspace",
            metadata={},
        )
        in events
    )
    assert events[-1] == AgentCompleted(text="The file says hello from the workspace.")
    assert provider.request_count == 2
    assert sessions.kinds == [
        "user_exchange",
        "assistant_exchange",
        "tool_started",
        "tool_finished",
        "tool_continuation",
        "assistant_exchange",
        "turn_completed",
    ]
    continuation = sessions.records[4].payload
    assert continuation.results[0].tool_use_id == "call-1"
    assert continuation.results[0].is_error is False
    assert "1: hello from the workspace" in continuation.results[0].content


@pytest.mark.asyncio
async def test_read_file_escape_is_a_recoverable_tool_result(workspace: Path) -> None:
    provider = FakeProvider(
        responses=[
            AssistantExchange(
                blocks=(
                    ToolUseBlock(
                        call_id="call-escape",
                        name="read_file",
                        input={"path": "../outside.txt"},
                    ),
                ),
                stop_reason="tool_use",
            ),
            AssistantExchange(
                blocks=(TextBlock(text="I could not read outside the workspace."),),
                stop_reason="end_turn",
            ),
        ]
    )
    sessions = InMemorySessionStore()
    catalog = await ToolCatalog.create([BuiltinToolSource(workspace)])
    application = AgentApplication(
        provider=provider,
        dispatcher=ToolDispatcher(catalog),
        sessions=sessions,
    )

    events = [event async for event in application.run("Read outside.txt")]

    assert any(
        isinstance(event, ToolFinished)
        and event.call_id == "call-escape"
        and event.is_error
        and "outside the workspace" in event.content.lower()
        for event in events
    )
    continuation = sessions.records[4].payload
    assert continuation.results[0].is_error is True
    assert "outside the workspace" in continuation.results[0].content.lower()
    assert events[-1] == AgentCompleted(text="I could not read outside the workspace.")


@pytest.mark.asyncio
async def test_multiple_tool_calls_keep_ids_and_results_in_order(workspace: Path) -> None:
    (workspace / "one.txt").write_text("one", encoding="utf-8")
    (workspace / "two.txt").write_text("two", encoding="utf-8")
    assistant = AssistantExchange(
        (
            ToolUseBlock("call-1", "read_file", {"path": "one.txt"}),
            ToolUseBlock("call-2", "read_file", {"path": "two.txt"}),
        ),
        "tool_use",
    )
    provider = FakeProvider([assistant, AssistantExchange((TextBlock("both read"),), "end_turn")])
    sessions = InMemorySessionStore()
    catalog = await ToolCatalog.create([BuiltinToolSource(workspace)])
    application = AgentApplication(provider, ToolDispatcher(catalog), sessions)

    _ = [event async for event in application.run("read both")]

    continuation = next(
        record.payload for record in sessions.records if record.kind == "tool_continuation"
    )
    assert isinstance(continuation, ToolContinuationExchange)
    assert continuation.assistant == assistant
    assert tuple(result.tool_use_id for result in continuation.results) == ("call-1", "call-2")
    assert tuple(result.content for result in continuation.results) == ("1: one", "1: two")


@pytest.mark.asyncio
async def test_repeated_identical_tool_results_warn_the_model_then_stop(workspace: Path) -> None:
    (workspace / "unchanged.txt").write_text("same", encoding="utf-8")
    provider = FakeProvider(
        [
            AssistantExchange(
                (
                    ToolUseBlock(f"call-{index}", "read_file", {"path": "unchanged.txt"}),
                    *(
                        (
                            ToolUseBlock(
                                "call-write",
                                "write_file",
                                {"path": "must-not-exist.txt", "content": "unsafe follow-up"},
                            ),
                        )
                        if index == 3
                        else ()
                    ),
                ),
                "tool_use",
            )
            for index in range(1, 4)
        ]
    )
    sessions = InMemorySessionStore()
    catalog = await ToolCatalog.create([BuiltinToolSource(workspace)])
    application = AgentApplication(provider, ToolDispatcher(catalog), sessions, max_steps=8)

    events = [event async for event in application.run("Keep reading until it changes")]

    warnings = [event for event in events if isinstance(event, WarningRaised)]
    assert len(warnings) == 1
    assert "same result twice" in warnings[0].message
    assert events[-1] == AgentFailed(
        message="Stopped after the same tool call produced the same result 3 times"
    )
    assert provider.request_count == 3
    third_request_continuation = provider.requests[2][-1]
    assert isinstance(third_request_continuation, ToolContinuationExchange)
    assert "NO_PROGRESS_WARNING" in third_request_continuation.results[0].content
    assert "no_progress_warning" in sessions.kinds
    assert "no_progress_stopped" in sessions.kinds
    assert sessions.kinds[-2:] == ["tool_continuation", "turn_failed"]
    assert not (workspace / "must-not-exist.txt").exists()
    final_continuation = next(
        record.payload
        for record in reversed(sessions.records)
        if record.kind == "tool_continuation"
    )
    assert [result.tool_use_id for result in final_continuation.results] == [
        "call-3",
        "call-write",
    ]
    assert final_continuation.results[1].metadata == {"skipped": True}


@pytest.mark.asyncio
async def test_duplicate_tool_names_are_rejected(workspace: Path) -> None:
    with pytest.raises(ValueError, match="duplicate tool name: read_file"):
        await ToolCatalog.create([BuiltinToolSource(workspace), BuiltinToolSource(workspace)])


@pytest.mark.asyncio
async def test_cancelling_a_running_tool_is_recorded_before_propagation(
    workspace: Path,
) -> None:
    class NoInput(BaseModel):
        pass

    class SlowTool:
        name = "slow"
        description = "Wait forever"
        input_model = NoInput

        async def execute(self, arguments: BaseModel) -> ToolOutput:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    class SlowSource:
        source_id = "test"

        async def list_tools(self):
            return (SlowTool(),)

    provider = FakeProvider(
        responses=[
            AssistantExchange((ToolUseBlock("call-slow", "slow", {}),), stop_reason="tool_use")
        ]
    )
    sessions = InMemorySessionStore()
    catalog = await ToolCatalog.create([SlowSource()])
    application = AgentApplication(provider, ToolDispatcher(catalog), sessions)
    events = application.run("wait")
    await anext(events)
    await anext(events)
    pending = asyncio.create_task(anext(events))
    await asyncio.sleep(0)

    pending.cancel()

    cancelled_tool = await pending
    cancelled_agent = await anext(events)
    assert isinstance(cancelled_tool, ToolFinished)
    assert cancelled_tool.status == "cancelled"
    assert isinstance(cancelled_agent, AgentCancelled)
    assert cancelled_agent.message == "Tool execution cancelled"
    assert sessions.kinds[-3:] == ["tool_cancelled", "tool_continuation", "turn_cancelled"]


@pytest.mark.asyncio
async def test_cancelling_a_provider_request_keeps_the_session_usable() -> None:
    class SlowProvider:
        async def stream(
            self,
            conversation: tuple[ConversationExchange, ...],
            tools: tuple[ToolSpec, ...],
            system_instructions: str | None = None,
        ) -> AsyncIterator[ProviderEvent]:
            del conversation, tools, system_instructions
            await asyncio.Event().wait()
            yield ProviderTextDelta("unreachable")

    sessions = InMemorySessionStore()
    application = AgentApplication(SlowProvider(), ToolDispatcher(ToolCatalog({})), sessions)
    events = application.run("wait")
    assert await anext(events)
    pending = asyncio.create_task(anext(events))
    await asyncio.sleep(0)

    pending.cancel()

    cancelled = await pending
    assert isinstance(cancelled, AgentCancelled)
    assert cancelled.message == "Provider request cancelled"
    assert sessions.kinds[-1] == "turn_cancelled"


@pytest.mark.asyncio
async def test_tool_display_events_are_redacted() -> None:
    class SecretInput(BaseModel):
        api_key: str

    class SecretTool:
        name = "secret_tool"
        description = "Return a secret for redaction testing"
        input_model = SecretInput

        async def execute(self, arguments: BaseModel) -> ToolOutput:
            parsed = SecretInput.model_validate(arguments)
            return ToolOutput(f"result={parsed.api_key}")

    provider = FakeProvider(
        [
            AssistantExchange(
                (ToolUseBlock("call-secret", "secret_tool", {"api_key": "secret-value"}),),
                "tool_use",
            ),
            AssistantExchange((TextBlock("done"),), "end_turn"),
        ]
    )
    catalog = ToolCatalog({"secret_tool": SecretTool()})
    application = AgentApplication(
        provider,
        ToolDispatcher(catalog),
        InMemorySessionStore(),
        display_redactor=Redactor(("secret-value",)).redact,
    )

    events = [event async for event in application.run("use tool")]

    started = next(event for event in events if isinstance(event, ToolStarted))
    finished = next(event for event in events if isinstance(event, ToolFinished))
    assert started.arguments == {"api_key": "[REDACTED]"}
    assert finished.content == "result=[REDACTED]"
