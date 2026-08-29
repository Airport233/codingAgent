from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent.application import AgentApplication
from coding_agent.domain import AssistantExchange, TextBlock, ToolUseBlock
from coding_agent.events import AgentCompleted, AgentFailed, AgentStarted
from coding_agent.providers.fake import FakeProvider
from coding_agent.sessions.memory import InMemorySessionStore
from coding_agent.skills import SkillLoader, SkillSnapshot, format_skill_list
from coding_agent.tools.catalog import ToolCatalog
from coding_agent.tools.dispatcher import ToolDispatcher
from coding_agent.tools.skills import SkillToolSource


def skill_snapshot() -> SkillSnapshot:
    loaded = SkillLoader.default().load()
    return SkillSnapshot(tuple(skill for skill in loaded.skills if skill.name == "test-fix"))


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
    assert "Run the narrowest command that reliably reproduces" in provider.system_instructions[0]


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
    assert "<available_skills>" in provider.system_instructions[0]
    assert "Reproduce the failure" not in provider.system_instructions[0]


@pytest.mark.asyncio
async def test_model_can_activate_a_disclosed_skill_for_the_next_step() -> None:
    snapshot = skill_snapshot()
    catalog = await ToolCatalog.create((SkillToolSource(snapshot),))
    provider = FakeProvider(
        [
            AssistantExchange(
                (ToolUseBlock("activate", "activate_skill", {"name": "test-fix"}),),
                "tool_use",
            ),
            AssistantExchange((TextBlock("fixed"),), "end_turn"),
        ]
    )
    sessions = InMemorySessionStore()
    application = AgentApplication(provider, ToolDispatcher(catalog), sessions, skills=snapshot)

    events = [event async for event in application.run("Fix the failing test")]

    assert events[-1] == AgentCompleted("fixed")
    assert "<available_skills>" in provider.system_instructions[0]
    assert "<active_skill" not in provider.system_instructions[0]
    assert "<active_skill" in provider.system_instructions[1]
    invoked = [record.payload for record in sessions.records if record.kind == "skill_invoked"]
    assert invoked == [
        {
            "name": "test-fix",
            "source": "builtin",
            "task": "Fix the failing test",
            "activation": "model",
        }
    ]


@pytest.mark.asyncio
async def test_lazily_corrupted_skill_body_fails_without_crashing_the_session(
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "skills" / "test-fix"
    skill_dir.mkdir(parents=True)
    entrypoint = skill_dir / "SKILL.md"
    entrypoint.write_text(
        "---\nname: test-fix\ndescription: Fix tests when they fail\n---\nvalid body\n",
        encoding="utf-8",
    )
    snapshot = SkillLoader(tmp_path / "skills").load()
    entrypoint.write_bytes(b"---\nname: test-fix\ndescription: Fix tests when they fail\n---\n\xff")
    application = AgentApplication(
        FakeProvider([]),
        ToolDispatcher(ToolCatalog({})),
        InMemorySessionStore(),
        skills=snapshot,
    )

    events = [event async for event in application.run("task", skill_name="test-fix")]

    assert isinstance(events[-1], AgentFailed)
    assert "Unable to load active skill" in events[-1].message


def test_application_exposes_skill_metadata_without_instructions() -> None:
    snapshot = skill_snapshot()
    snapshot = SkillSnapshot(snapshot.skills, ("Skipped skill broken.md: invalid",))
    application = AgentApplication(
        FakeProvider([]),
        ToolDispatcher(ToolCatalog({})),
        InMemorySessionStore(),
        skills=snapshot,
    )

    assert application.available_skills() == (
        (
            "test-fix",
            "Reproduce, diagnose, fix, and verify failing tests. Use when tests or CI fail, "
            "a regression needs a focused test, or the user asks to repair a broken test suite.",
            "builtin",
        ),
    )
    assert application.skill_warnings() == ("Skipped skill broken.md: invalid",)


def test_skill_listing_keeps_loader_warnings_when_no_skill_is_valid() -> None:
    rendered = format_skill_list((), ("Skipped skill broken.md: invalid",))

    assert "No coding workflows are available." in rendered
    assert "Skipped skill broken.md: invalid" in rendered
