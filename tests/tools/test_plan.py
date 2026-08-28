from __future__ import annotations

from typing import cast

import pytest

from coding_agent.domain import ToolUseBlock
from coding_agent.plan import PlanState, PlanStatus, PlanStep
from coding_agent.tools.catalog import ToolCatalog
from coding_agent.tools.dispatcher import ToolDispatcher
from coding_agent.tools.plan import UpdatePlanTool


@pytest.mark.asyncio
async def test_update_plan_tool_replaces_state_and_returns_structured_metadata() -> None:
    state = PlanState()
    dispatcher = ToolDispatcher(ToolCatalog({"update_plan": UpdatePlanTool(state)}))

    result = await dispatcher.execute(
        ToolUseBlock(
            "plan-1",
            "update_plan",
            {
                "explanation": "Start with the failing test",
                "plan": [
                    {"step": "Reproduce failure", "status": "in_progress"},
                    {"step": "Implement fix", "status": "pending"},
                ],
            },
        )
    )

    assert result.is_error is False
    assert state.snapshot() == (
        PlanStep("Reproduce failure", "in_progress"),
        PlanStep("Implement fix", "pending"),
    )
    assert result.metadata["explanation"] == "Start with the failing test"
    assert result.metadata["plan"] == [
        {"step": "Reproduce failure", "status": "in_progress"},
        {"step": "Implement fix", "status": "pending"},
    ]
    assert "1/2 active" in result.content


@pytest.mark.asyncio
async def test_update_plan_rejects_multiple_active_steps_without_mutating_state() -> None:
    state = PlanState((PlanStep("Existing", "pending"),))
    dispatcher = ToolDispatcher(ToolCatalog({"update_plan": UpdatePlanTool(state)}))

    result = await dispatcher.execute(
        ToolUseBlock(
            "plan-2",
            "update_plan",
            {
                "plan": [
                    {"step": "First", "status": "in_progress"},
                    {"step": "Second", "status": "in_progress"},
                ]
            },
        )
    )

    assert result.is_error is True
    assert "at most one" in result.content.lower()
    assert state.snapshot() == (PlanStep("Existing", "pending"),)


@pytest.mark.asyncio
async def test_update_plan_limits_size_and_rejects_duplicate_steps() -> None:
    state = PlanState()
    dispatcher = ToolDispatcher(ToolCatalog({"update_plan": UpdatePlanTool(state)}))

    duplicate = await dispatcher.execute(
        ToolUseBlock(
            "plan-3",
            "update_plan",
            {
                "plan": [
                    {"step": "Run tests", "status": "pending"},
                    {"step": "Run tests", "status": "completed"},
                ]
            },
        )
    )
    oversized = await dispatcher.execute(
        ToolUseBlock(
            "plan-4",
            "update_plan",
            {"plan": [{"step": f"Step {index}", "status": "pending"} for index in range(9)]},
        )
    )

    assert duplicate.is_error is True
    assert oversized.is_error is True
    assert state.snapshot() == ()


def test_plan_state_can_restore_valid_json_metadata() -> None:
    state = PlanState()

    state.restore(
        [
            {"step": "Inspect", "status": "completed"},
            {"step": "Verify", "status": "in_progress"},
        ]
    )

    assert state.snapshot() == (
        PlanStep("Inspect", "completed"),
        PlanStep("Verify", "in_progress"),
    )
    assert state.render_context().startswith("<current_plan>")
    assert "[in_progress] Verify" in state.render_context()


def test_plan_state_direct_update_is_normalized_and_atomic() -> None:
    state = PlanState((PlanStep("  Inspect  ", "pending"),))
    assert state.snapshot() == (PlanStep("Inspect", "pending"),)

    with pytest.raises(ValueError, match="status"):
        state.update((PlanStep("Invalid", cast(PlanStatus, "blocked")),))

    assert state.snapshot() == (PlanStep("Inspect", "pending"),)
