from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from coding_agent.application import AgentApplication
from coding_agent.domain import AssistantExchange, TextBlock, ToolUseBlock
from coding_agent.events import AgentCompleted, ToolFinished
from coding_agent.providers.fake import FakeProvider
from coding_agent.sessions.memory import InMemorySessionStore
from coding_agent.tools.builtin import BuiltinToolSource
from coding_agent.tools.catalog import ToolCatalog
from coding_agent.tools.dispatcher import ToolDispatcher


@pytest.mark.asyncio
async def test_agent_can_inspect_existing_git_changes_before_finishing(tmp_path: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is unavailable")
    subprocess.run([git, "init", "-q", str(tmp_path)], check=True)
    subprocess.run([git, "-C", str(tmp_path), "config", "user.name", "Test User"], check=True)
    subprocess.run(
        [git, "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    subprocess.run([git, "-C", str(tmp_path), "add", "app.py"], check=True)
    subprocess.run([git, "-C", str(tmp_path), "commit", "-qm", "initial"], check=True)
    target.write_text("value = 2\n", encoding="utf-8")
    provider = FakeProvider(
        [
            AssistantExchange(
                (ToolUseBlock("status", "git_status", {}),),
                "tool_use",
            ),
            AssistantExchange(
                (
                    ToolUseBlock(
                        "diff",
                        "git_diff",
                        {"scope": "unstaged", "path": "app.py"},
                    ),
                ),
                "tool_use",
            ),
            AssistantExchange(
                (TextBlock("The existing app.py change sets value to 2."),), "end_turn"
            ),
        ]
    )
    catalog = await ToolCatalog.create((BuiltinToolSource(tmp_path),))
    application = AgentApplication(
        provider, ToolDispatcher(catalog), InMemorySessionStore(), max_steps=4
    )

    events = [event async for event in application.run("Inspect my current changes")]

    status = next(
        event
        for event in events
        if isinstance(event, ToolFinished) and event.tool_name == "git_status"
    )
    diff = next(
        event
        for event in events
        if isinstance(event, ToolFinished) and event.tool_name == "git_diff"
    )
    assert "app.py" in status.content
    assert "+value = 2" in diff.content
    assert events[-1] == AgentCompleted("The existing app.py change sets value to 2.")
