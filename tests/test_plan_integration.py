from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent.application import AgentApplication
from coding_agent.domain import AssistantExchange, TextBlock, ToolUseBlock
from coding_agent.events import AgentCompleted, PlanUpdated
from coding_agent.plan import PlanState, PlanStep
from coding_agent.providers.fake import FakeProvider
from coding_agent.runtime import RuntimeSettings, create_runtime
from coding_agent.sessions.jsonl import JsonlSessionRepository
from coding_agent.sessions.memory import InMemorySessionStore
from coding_agent.tools.builtin import BuiltinToolSource
from coding_agent.tools.catalog import ToolCatalog
from coding_agent.tools.dispatcher import ToolDispatcher
from coding_agent.tools.plan import UpdatePlanTool


@pytest.mark.asyncio
async def test_plan_update_is_persisted_emitted_and_injected_into_next_request() -> None:
    state = PlanState()
    provider = FakeProvider(
        [
            AssistantExchange(
                (
                    ToolUseBlock(
                        "plan-call",
                        "update_plan",
                        {
                            "explanation": "Track the fix",
                            "plan": [
                                {"step": "Inspect", "status": "completed"},
                                {"step": "Fix", "status": "in_progress"},
                                {"step": "Verify", "status": "pending"},
                            ],
                        },
                    ),
                ),
                "tool_use",
            ),
            AssistantExchange((TextBlock("working from the plan"),), "end_turn"),
        ]
    )
    sessions = InMemorySessionStore()
    application = AgentApplication(
        provider,
        ToolDispatcher(ToolCatalog({"update_plan": UpdatePlanTool(state)})),
        sessions,
        plan_state=state,
    )

    events = [event async for event in application.run("Fix the bug")]

    assert (
        PlanUpdated(
            (
                PlanStep("Inspect", "completed"),
                PlanStep("Fix", "in_progress"),
                PlanStep("Verify", "pending"),
            ),
            "Track the fix",
        )
        in events
    )
    assert events[-1] == AgentCompleted("working from the plan")
    assert "plan_updated" in sessions.kinds
    assert "<current_plan>" in provider.system_instructions[1]
    assert "[in_progress] Fix" in provider.system_instructions[1]
    assert application.current_plan() == state.snapshot()


@pytest.mark.asyncio
async def test_jsonl_resume_restores_latest_valid_plan(tmp_path: Path) -> None:
    repository = JsonlSessionRepository(tmp_path / "data")
    store = await repository.create(tmp_path, session_id="planned")
    await store.append("user_exchange", {"type": "user_exchange", "content": "task"})
    await store.append(
        "plan_updated",
        {
            "explanation": "Latest plan",
            "plan": [
                {"step": "Inspect", "status": "completed"},
                {"step": "Verify", "status": "in_progress"},
            ],
        },
    )

    recovered = await repository.resume(tmp_path, "planned")

    assert recovered is not None
    assert recovered.plan == (
        PlanStep("Inspect", "completed"),
        PlanStep("Verify", "in_progress"),
    )


@pytest.mark.asyncio
async def test_builtin_catalog_exposes_update_plan(tmp_path: Path) -> None:
    state = PlanState()

    tools = await BuiltinToolSource(tmp_path, plan_state=state).list_tools()

    assert "update_plan" in {tool.name for tool in tools}


@pytest.mark.asyncio
async def test_runtime_resume_shares_recovered_plan_with_application_and_tool(
    tmp_path: Path,
) -> None:
    settings = RuntimeSettings.from_environment(
        workspace=tmp_path,
        model="example-model",
        environ={
            "CODING_AGENT_BASE_URL": "https://example.invalid/anthropic",
            "CODING_AGENT_API_KEY": "private-test-credential",
        },
        data_root=tmp_path / "data",
    )
    provider = FakeProvider(
        [
            AssistantExchange(
                (
                    ToolUseBlock(
                        "plan-runtime",
                        "update_plan",
                        {"plan": [{"step": "Verify", "status": "in_progress"}]},
                    ),
                ),
                "tool_use",
            ),
            AssistantExchange((TextBlock("planned"),), "end_turn"),
        ]
    )
    initial = await create_runtime(settings, provider=provider)
    _ = [event async for event in initial.application.run("Plan this task")]
    session_id = initial.session_id
    await initial.aclose()

    resumed = await create_runtime(
        settings, resume_session_id=session_id, provider=FakeProvider([])
    )

    assert resumed.application.current_plan() == (PlanStep("Verify", "in_progress"),)
