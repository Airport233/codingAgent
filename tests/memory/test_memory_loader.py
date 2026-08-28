from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from coding_agent.application import AgentApplication
from coding_agent.domain import AssistantExchange, TextBlock
from coding_agent.memory.loader import MemoryLoadError, ProjectMemoryLoader
from coding_agent.providers.fake import FakeProvider
from coding_agent.sessions.memory import InMemorySessionStore
from coding_agent.tools.builtin import BuiltinToolSource
from coding_agent.tools.catalog import ToolCatalog
from coding_agent.tools.dispatcher import ToolDispatcher


@pytest.fixture
def workspace() -> Path:
    path = Path.cwd() / ".tmp" / "memory"
    shutil.rmtree(path, ignore_errors=True)
    (path / "src/package").mkdir(parents=True)
    (path / "sibling").mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_memory_loads_from_project_root_to_cwd_with_nearest_priority(
    workspace: Path,
) -> None:
    (workspace / "CODING_AGENT.md").write_text("root rule", encoding="utf-8")
    (workspace / "src/CODING_AGENT.md").write_text("src rule", encoding="utf-8")
    (workspace / "src/package/CODING_AGENT.md").write_text("package rule", encoding="utf-8")
    (workspace / "sibling/CODING_AGENT.md").write_text("ignore me", encoding="utf-8")
    loader = ProjectMemoryLoader(workspace, workspace / "src/package")

    snapshot = loader.load()

    assert [entry.source for entry in snapshot.entries] == [
        "CODING_AGENT.md",
        "src/CODING_AGENT.md",
        "src/package/CODING_AGENT.md",
    ]
    assert [entry.priority for entry in snapshot.entries] == [0, 1, 2]
    assert snapshot.rendered.index("root rule") < snapshot.rendered.index("package rule")
    assert "later sections have higher priority" in snapshot.rendered
    assert "ignore me" not in snapshot.rendered


def test_memory_is_reloaded_and_digest_changes_with_file_content(workspace: Path) -> None:
    memory = workspace / "CODING_AGENT.md"
    memory.write_text("first", encoding="utf-8")
    loader = ProjectMemoryLoader(workspace, workspace)

    first = loader.load()
    memory.write_text("second", encoding="utf-8")
    second = loader.load()

    assert first.digest != second.digest
    assert "first" in first.rendered
    assert "second" in second.rendered


def test_memory_rejects_outside_binary_and_oversized_content(workspace: Path) -> None:
    with pytest.raises(MemoryLoadError, match="inside the project"):
        ProjectMemoryLoader(workspace, workspace.parent).load()

    memory = workspace / "CODING_AGENT.md"
    memory.write_bytes(b"text\x00binary")
    with pytest.raises(MemoryLoadError, match="binary"):
        ProjectMemoryLoader(workspace, workspace).load()

    memory.write_text("12345", encoding="utf-8")
    with pytest.raises(MemoryLoadError, match="too large"):
        ProjectMemoryLoader(workspace, workspace, max_file_bytes=4).load()


def test_memory_symlink_cannot_escape_the_project(workspace: Path) -> None:
    outside = workspace.parent / "outside-memory.md"
    outside.write_text("outside", encoding="utf-8")
    link = workspace / "CODING_AGENT.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(MemoryLoadError, match="outside the project"):
        ProjectMemoryLoader(workspace, workspace).load()


@pytest.mark.asyncio
async def test_agent_refreshes_memory_between_user_turns(workspace: Path) -> None:
    memory = workspace / "CODING_AGENT.md"
    memory.write_text("first rule", encoding="utf-8")
    provider = FakeProvider(
        [
            AssistantExchange((TextBlock("one"),), "end_turn"),
            AssistantExchange((TextBlock("two"),), "end_turn"),
        ]
    )
    sessions = InMemorySessionStore()
    catalog = await ToolCatalog.create([BuiltinToolSource(workspace)])
    application = AgentApplication(
        provider,
        ToolDispatcher(catalog),
        sessions,
        memory_loader=ProjectMemoryLoader(workspace, workspace),
    )

    _ = [event async for event in application.run("first turn")]
    memory.write_text("second rule", encoding="utf-8")
    _ = [event async for event in application.run("second turn")]

    assert "first rule" in provider.system_instructions[0]
    assert "second rule" in provider.system_instructions[1]
    assert sessions.kinds.count("memory_snapshot_changed") == 2
