from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from coding_agent.context import ContextBudget, ContextManager, TokenEstimator
from coding_agent.domain import (
    AssistantExchange,
    TextBlock,
    ToolContinuationExchange,
    ToolResultBlock,
    ToolUseBlock,
    UserExchange,
)


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
    assert "Earlier conversation summary" in candidate.projected[0].content
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
    assert store.records[0][0] == "compaction_completed"

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
