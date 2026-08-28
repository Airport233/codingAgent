from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from pydantic import BaseModel

from coding_agent.application import AgentApplication
from coding_agent.domain import AssistantExchange, TextBlock, ToolUseBlock
from coding_agent.events import AgentCompleted, TextDelta, ToolFinished
from coding_agent.providers.fake import FakeProvider
from coding_agent.sessions.memory import InMemorySessionStore
from coding_agent.tools.base import ToolOutput
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
    assert ToolFinished(call_id="call-1", tool_name="read_file", is_error=False) in events
    assert events[-1] == AgentCompleted(text="The file says hello from the workspace.")
    assert provider.request_count == 2
    assert sessions.kinds == [
        "user_exchange",
        "assistant_exchange",
        "tool_started",
        "tool_finished",
        "tool_continuation",
        "assistant_exchange",
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

    assert ToolFinished(call_id="call-escape", tool_name="read_file", is_error=True) in events
    continuation = sessions.records[4].payload
    assert continuation.results[0].is_error is True
    assert "outside the workspace" in continuation.results[0].content.lower()
    assert events[-1] == AgentCompleted(text="I could not read outside the workspace.")


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

    with pytest.raises(asyncio.CancelledError):
        await pending
    assert sessions.kinds[-2:] == ["tool_cancelled", "turn_cancelled"]
