from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent.application import AgentApplication
from coding_agent.domain import AssistantExchange, TextBlock
from coding_agent.events import AgentCompleted, AgentFailed, AgentStarted
from coding_agent.providers.fake import FakeProvider
from coding_agent.sessions.memory import InMemorySessionStore
from coding_agent.skills import SkillDefinition, SkillSnapshot, format_skill_list
from coding_agent.tools.catalog import ToolCatalog
from coding_agent.tools.dispatcher import ToolDispatcher


def skill_snapshot() -> SkillSnapshot:
    return SkillSnapshot(
        (
            SkillDefinition(
                name="test-fix",
                description="Fix a failing test",
                instructions="Reproduce the failure before editing.",
                source="builtin",
                path=Path("test-fix.md"),
            ),
        )
    )


@pytest.mark.asyncio
async def test_skill_is_injected_for_one_turn_and_persisted_for_audit() -> None:
    provider = FakeProvider([AssistantExchange((TextBlock("fixed"),), "end_turn")])
    sessions = InMemorySessionStore()
    application = AgentApplication(
        provider,
        ToolDispatcher(ToolCatalog({})),
        sessions,
        skills=skill_snapshot(),
    )

    events = [
        event async for event in application.run("Fix the parser regression", skill_name="test-fix")
    ]

    assert events[0] == AgentStarted("Fix the parser regression", skill_name="test-fix")
    assert events[-1] == AgentCompleted("fixed")
    assert sessions.kinds[:2] == ["skill_invoked", "user_exchange"]
    assert sessions.records[0].payload == {
        "name": "test-fix",
        "source": "builtin",
        "task": "Fix the parser regression",
    }
    assert '<active_skill name="test-fix" source="builtin">' in provider.system_instructions[0]
    assert "Reproduce the failure before editing." in provider.system_instructions[0]


@pytest.mark.asyncio
async def test_unknown_skill_fails_without_mutating_conversation_or_session() -> None:
    sessions = InMemorySessionStore()
    application = AgentApplication(
        FakeProvider([]),
        ToolDispatcher(ToolCatalog({})),
        sessions,
        skills=skill_snapshot(),
    )

    events = [event async for event in application.run("task", skill_name="missing")]

    assert events == [AgentFailed("Unknown skill: missing")]
    assert application.conversation_history() == ()
    assert sessions.records == []


@pytest.mark.asyncio
async def test_ordinary_turn_does_not_inject_any_skill() -> None:
    provider = FakeProvider([AssistantExchange((TextBlock("normal"),), "end_turn")])
    application = AgentApplication(
        provider,
        ToolDispatcher(ToolCatalog({})),
        InMemorySessionStore(),
        skills=skill_snapshot(),
    )

    _ = [event async for event in application.run("ordinary task")]

    assert "active_skill" not in provider.system_instructions[0]


def test_application_exposes_skill_metadata_without_instructions() -> None:
    snapshot = skill_snapshot()
    snapshot = SkillSnapshot(snapshot.skills, ("Skipped skill broken.md: invalid",))
    application = AgentApplication(
        FakeProvider([]),
        ToolDispatcher(ToolCatalog({})),
        InMemorySessionStore(),
        skills=snapshot,
    )

    assert application.available_skills() == (("test-fix", "Fix a failing test", "builtin"),)
    assert application.skill_warnings() == ("Skipped skill broken.md: invalid",)


def test_skill_listing_keeps_loader_warnings_when_no_skill_is_valid() -> None:
    rendered = format_skill_list((), ("Skipped skill broken.md: invalid",))

    assert "No coding workflows are available." in rendered
    assert "Skipped skill broken.md: invalid" in rendered
