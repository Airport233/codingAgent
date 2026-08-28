from __future__ import annotations

import pytest

from coding_agent.application import AgentApplication
from coding_agent.approval import ConfigurableApprovalPolicy
from coding_agent.domain import AssistantExchange, ToolResultBlock, ToolUseBlock
from coding_agent.events import AgentFailed, WarningRaised
from coding_agent.no_progress import NoProgressDetector
from coding_agent.providers.fake import FakeProvider
from coding_agent.sessions.memory import InMemorySessionStore
from coding_agent.tools.catalog import ToolCatalog
from coding_agent.tools.dispatcher import ToolDispatcher


def test_detector_normalizes_argument_order_and_warns_then_stops() -> None:
    detector = NoProgressDetector()
    first = ToolUseBlock("call-1", "code_search", {"query": "TODO", "path": "src"})
    reordered = ToolUseBlock("call-2", "code_search", {"path": "src", "query": "TODO"})

    assert detector.observe(first, ToolResultBlock("call-1", "same", False)).action == "none"
    warning = detector.observe(reordered, ToolResultBlock("call-2", "same", False))
    stopped = detector.observe(first, ToolResultBlock("call-1", "same", False))

    assert warning.action == "warn"
    assert warning.repetition_count == 2
    assert stopped.action == "stop"
    assert stopped.repetition_count == 3
    assert len(stopped.fingerprint) == 16


def test_detector_resets_when_the_tool_result_changes() -> None:
    detector = NoProgressDetector()
    call = ToolUseBlock("call", "read_file", {"path": "src/app.py"})

    assert detector.observe(call, ToolResultBlock("call", "version one", False)).action == "none"
    assert detector.observe(call, ToolResultBlock("call", "version two", False)).action == "none"
    assert detector.observe(call, ToolResultBlock("call", "version two", False)).action == "warn"


def test_detector_includes_error_state_in_progress_fingerprint() -> None:
    detector = NoProgressDetector()
    call = ToolUseBlock("call", "shell", {"command": "pytest -q"})

    detector.observe(call, ToolResultBlock("call", "exit 1", True))
    observation = detector.observe(call, ToolResultBlock("call", "exit 1", False))

    assert observation.action == "none"
    assert observation.repetition_count == 1


def test_detector_ignores_shell_duration_but_not_command_output() -> None:
    detector = NoProgressDetector()
    call = ToolUseBlock("call", "shell", {"command": "pytest -q"})

    first = detector.observe(
        call,
        ToolResultBlock("call", "exit_code: 1\nduration_ms: 12\nstdout:\nfailed", True),
    )
    repeated = detector.observe(
        call,
        ToolResultBlock("call", "exit_code: 1\nduration_ms: 99\nstdout:\nfailed", True),
    )
    changed = detector.observe(
        call,
        ToolResultBlock("call", "exit_code: 0\nduration_ms: 15\nstdout:\npassed", False),
    )

    assert first.action == "none"
    assert repeated.action == "warn"
    assert changed.action == "none"


@pytest.mark.parametrize("warn_at, stop_at", [(1, 3), (2, 2), (4, 3)])
def test_detector_rejects_invalid_thresholds(warn_at: int, stop_at: int) -> None:
    with pytest.raises(ValueError, match="thresholds"):
        NoProgressDetector(warn_at=warn_at, stop_at=stop_at)


@pytest.mark.asyncio
async def test_denied_tool_loop_is_also_stopped_without_execution() -> None:
    provider = FakeProvider(
        [
            AssistantExchange(
                (ToolUseBlock(f"call-{index}", "shell", {"command": "dangerous"}),),
                "tool_use",
            )
            for index in range(3)
        ]
    )
    sessions = InMemorySessionStore()
    application = AgentApplication(
        provider,
        ToolDispatcher(ToolCatalog({})),
        sessions,
        approval_policy=ConfigurableApprovalPolicy(mode="deny", guarded_tools=frozenset({"shell"})),
    )

    events = [event async for event in application.run("Do not loop")]

    assert sum(isinstance(event, WarningRaised) for event in events) == 1
    assert isinstance(events[-1], AgentFailed)
    assert "3 times" in events[-1].message
    assert sessions.kinds.count("no_progress_warning") == 1
    assert sessions.kinds.count("no_progress_stopped") == 1
