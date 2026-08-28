from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from coding_agent.application import AgentApplication
from coding_agent.context import ContextBudget, ContextManager, TokenEstimator
from coding_agent.domain import (
    AssistantExchange,
    ConversationExchange,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolContinuationExchange,
    ToolResultBlock,
    ToolUseBlock,
    UserExchange,
)
from coding_agent.events import AgentFailed
from coding_agent.providers.fake import FakeProvider
from coding_agent.sessions.memory import InMemorySessionStore
from coding_agent.tools.catalog import ToolCatalog
from coding_agent.tools.dispatcher import ToolDispatcher


def exchanges():
    tool_assistant = AssistantExchange(
        (ToolUseBlock("call-1", "read_file", {"path": "a.py"}),),
        "tool_use",
    )
    return (
        UserExchange("old task " * 30),
        AssistantExchange((TextBlock("old answer " * 30),), "end_turn"),
        ToolContinuationExchange(
            tool_assistant,
            (ToolResultBlock("call-1", "1: content", False),),
        ),
        UserExchange("recent question"),
        AssistantExchange((TextBlock("recent answer"),), "end_turn"),
    )


def test_budget_reports_soft_and_hard_pressure() -> None:
    budget = ContextBudget(context_window=100, max_output_tokens=20, auto_ratio=0.5)

    assert budget.status(39).level == "safe"
    assert budget.status(40).level == "soft"
    assert budget.status(80).level == "hard"


def test_provider_usage_calibrates_future_estimates_without_claiming_exact_status() -> None:
    history = (UserExchange("hello world"),)
    estimator = TokenEstimator()
    manager = ContextManager(
        ContextBudget(context_window=100, max_output_tokens=20),
        estimator,
    )
    before = manager.status(history).used_tokens

    manager.record_provider_usage(history, {"input_tokens": 20})
    status = manager.status(history)

    assert status.used_tokens > before
    assert status.usage_source == "estimated"
    assert status.last_provider_input_tokens == 20


def test_model_change_projection_removes_thinking_without_mutating_raw_history() -> None:
    tool_assistant = AssistantExchange(
        (
            ThinkingBlock("private tool reasoning", signature="signed"),
            ToolUseBlock("call-1", "read_file", {"path": "a.py"}),
        ),
        "tool_use",
    )
    history = (
        UserExchange("inspect the file"),
        AssistantExchange(
            (
                ThinkingBlock("private reasoning", signature="signed"),
                RedactedThinkingBlock("opaque"),
                TextBlock("public answer"),
            ),
            "end_turn",
        ),
        ToolContinuationExchange(
            tool_assistant,
            (ToolResultBlock("call-1", "contents", False),),
        ),
    )
    manager = ContextManager(
        ContextBudget(context_window=1_000, max_output_tokens=100),
        TokenEstimator(),
        excluded_thinking_indices=frozenset({1, 2}),
    )

    projected = manager.project(history)

    assert history[1].blocks[0] == ThinkingBlock("private reasoning", signature="signed")
    assistant = projected[1]
    assert isinstance(assistant, AssistantExchange)
    assert assistant.blocks == (TextBlock("public answer"),)
    continuation = projected[2]
    assert isinstance(continuation, ToolContinuationExchange)
    assert continuation.assistant.blocks == (ToolUseBlock("call-1", "read_file", {"path": "a.py"}),)


def test_deterministic_compaction_keeps_complete_recent_exchanges() -> None:
    manager = ContextManager(
        ContextBudget(context_window=160, max_output_tokens=20, auto_ratio=0.5),
        TokenEstimator(),
        retained_exchanges=2,
    )
    history = exchanges()

    candidate = manager.prepare(history, reason="manual")

    assert candidate is not None
    assert candidate.retained_from == 3
    assert candidate.projected[-2:] == history[-2:]
    assert isinstance(candidate.projected[0], UserExchange)
    assert "context_summary_version: 1" in candidate.projected[0].content
    assert all(not isinstance(item, ToolContinuationExchange) for item in candidate.projected[:1])


@dataclass
class RecordingStore:
    records: list[tuple[str, object]] = field(default_factory=list)
    fail: bool = False

    async def append(self, kind: str, payload: object) -> None:
        if self.fail:
            raise OSError("disk unavailable")
        self.records.append((kind, payload))


@pytest.mark.asyncio
async def test_compaction_installs_only_after_checkpoint_is_durable() -> None:
    manager = ContextManager(
        ContextBudget(context_window=160, max_output_tokens=20, auto_ratio=0.5),
        TokenEstimator(),
        retained_exchanges=2,
    )
    history = exchanges()
    store = RecordingStore()

    checkpoint = await manager.compact(history, reason="manual", persist=store.append)

    assert checkpoint is not None
    assert manager.project(history) == checkpoint.projected
    assert [kind for kind, _payload in store.records] == [
        "compaction_started",
        "compaction_completed",
    ]

    failing_manager = ContextManager(
        ContextBudget(context_window=160, max_output_tokens=20, auto_ratio=0.5),
        TokenEstimator(),
        retained_exchanges=2,
    )
    with pytest.raises(OSError, match="disk unavailable"):
        await failing_manager.compact(
            history,
            reason="manual",
            persist=RecordingStore(fail=True).append,
        )
    assert failing_manager.project(history) == history


@pytest.mark.asyncio
async def test_invalid_provider_summary_uses_independent_deterministic_fallback() -> None:
    manager = ContextManager(
        ContextBudget(context_window=160, max_output_tokens=20, auto_ratio=0.5),
        TokenEstimator(),
        retained_exchanges=2,
    )
    store = RecordingStore()

    async def invalid_summary(_history: tuple[ConversationExchange, ...]) -> str:
        return "not a structured summary"

    checkpoint = await manager.compact(
        exchanges(),
        reason="manual",
        persist=store.append,
        summarize=invalid_summary,
    )

    assert checkpoint is not None
    assert "strategy: deterministic" in checkpoint.summary
    assert [kind for kind, _payload in store.records] == [
        "compaction_started",
        "compaction_strategy_failed",
        "compaction_completed",
    ]
    completed = store.records[-1][1]
    assert isinstance(completed, dict)
    assert completed["strategy"] == "deterministic"


def test_checkpoint_can_be_restored_against_replayed_history() -> None:
    manager = ContextManager(
        ContextBudget(context_window=160, max_output_tokens=20, auto_ratio=0.5),
        TokenEstimator(),
        retained_exchanges=2,
    )
    history = exchanges()

    restored = manager.restore(
        history,
        {
            "reason": "manual",
            "retained_from": 3,
            "before_tokens": 250,
            "after_tokens": 70,
            "summary": "Earlier conversation summary: fixed the parser.",
        },
    )

    assert manager.project(history) == restored.projected
    assert restored.projected[-2:] == history[-2:]

    with pytest.raises(ValueError, match="Invalid compaction checkpoint"):
        manager.restore(history, {"retained_from": 99})


@pytest.mark.asyncio
async def test_agent_auto_compacts_before_request_without_mutating_raw_history() -> None:
    history = exchanges()
    provider = FakeProvider(
        [
            AssistantExchange(
                (
                    TextBlock(
                        "task_goal: finish the task\n"
                        "user_constraints: keep changes focused\n"
                        "decisions: inspect before editing\n"
                        "files_read: a.py\n"
                        "files_modified: none\n"
                        "commands_and_results: read_file succeeded\n"
                        "verification_status: pending\n"
                        "known_failures: none\n"
                        "pending_work: answer the latest request"
                    ),
                ),
                stop_reason="end_turn",
            ),
            AssistantExchange((TextBlock("finished"),), stop_reason="end_turn"),
        ]
    )
    sessions = InMemorySessionStore()
    application = AgentApplication(
        provider,
        ToolDispatcher(ToolCatalog({})),
        sessions,
        initial_exchanges=history,
        context_manager=ContextManager(
            ContextBudget(context_window=400, max_output_tokens=20, auto_ratio=0.5),
            TokenEstimator(),
            retained_exchanges=2,
        ),
    )

    _ = [event async for event in application.run("new request " * 20)]

    assert "compaction_completed" in sessions.kinds
    assert provider.request_count == 2
    assert isinstance(provider.requests[1][0], UserExchange)
    assert "task_goal: finish the task" in provider.requests[1][0].content
    assert len(provider.requests[1]) < len(history) + 1


@pytest.mark.asyncio
async def test_agent_refuses_provider_request_when_hard_limit_cannot_be_compacted() -> None:
    provider = FakeProvider(
        [AssistantExchange((TextBlock("must not be requested"),), stop_reason="end_turn")]
    )
    application = AgentApplication(
        provider,
        ToolDispatcher(ToolCatalog({})),
        InMemorySessionStore(),
        initial_exchanges=(UserExchange("uncompactable history " * 30),),
        context_manager=ContextManager(
            ContextBudget(context_window=100, max_output_tokens=20),
            TokenEstimator(),
        ),
    )

    events = [event async for event in application.run("new request " * 10)]

    assert provider.request_count == 0
    assert isinstance(events[-1], AgentFailed)
    assert events[-1].message == "Context remains above the safe request limit"
