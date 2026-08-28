from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from coding_agent.domain import ToolUseBlock
from coding_agent.tools.builtin import BuiltinToolSource
from coding_agent.tools.catalog import ToolCatalog
from coding_agent.tools.dispatcher import ToolDispatcher
from coding_agent.tools.git import GitDiffTool, GitStatusTool


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is unavailable")
    subprocess.run([git, "init", "-q", str(tmp_path)], check=True)
    subprocess.run([git, "-C", str(tmp_path), "config", "user.name", "Test User"], check=True)
    subprocess.run(
        [git, "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    (tmp_path / "tracked.txt").write_text("before\n", encoding="utf-8")
    subprocess.run([git, "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run([git, "-C", str(tmp_path), "commit", "-qm", "initial"], check=True)
    return tmp_path


@pytest.mark.asyncio
async def test_git_status_reports_tracked_and_untracked_changes(repository: Path) -> None:
    (repository / "tracked.txt").write_text("after\n", encoding="utf-8")
    (repository / "new.txt").write_text("new\n", encoding="utf-8")
    dispatcher = ToolDispatcher(ToolCatalog({"git_status": GitStatusTool(repository)}))

    result = await dispatcher.execute(ToolUseBlock("status", "git_status", {}))

    assert result.is_error is False
    assert "tracked.txt" in result.content
    assert "new.txt" in result.content
    assert result.metadata["truncated"] is False


@pytest.mark.asyncio
async def test_git_diff_supports_unstaged_staged_and_path_filter(repository: Path) -> None:
    (repository / "tracked.txt").write_text("after\n", encoding="utf-8")
    dispatcher = ToolDispatcher(ToolCatalog({"git_diff": GitDiffTool(repository)}))

    unstaged = await dispatcher.execute(
        ToolUseBlock(
            "diff-1",
            "git_diff",
            {"scope": "unstaged", "path": "tracked.txt", "context_lines": 1},
        )
    )
    subprocess.run(
        [shutil.which("git") or "git", "-C", str(repository), "add", "tracked.txt"],
        check=True,
    )
    staged = await dispatcher.execute(ToolUseBlock("diff-2", "git_diff", {"scope": "staged"}))

    assert unstaged.is_error is False
    assert "+after" in unstaged.content
    assert "tracked.txt" in unstaged.content
    assert staged.is_error is False
    assert "+after" in staged.content
    assert staged.metadata["scope"] == "staged"


@pytest.mark.asyncio
async def test_git_tools_return_recoverable_errors_for_non_repo_and_path_escape(
    tmp_path: Path,
) -> None:
    dispatcher = ToolDispatcher(
        ToolCatalog(
            {
                "git_status": GitStatusTool(tmp_path),
                "git_diff": GitDiffTool(tmp_path),
            }
        )
    )

    not_repo = await dispatcher.execute(ToolUseBlock("status", "git_status", {}))
    escaped = await dispatcher.execute(ToolUseBlock("diff", "git_diff", {"path": "../outside.py"}))

    assert not_repo.is_error is True
    assert "git repository" in not_repo.content.lower()
    assert escaped.is_error is True
    assert "workspace" in escaped.content.lower()


@pytest.mark.asyncio
async def test_git_diff_truncates_large_output_with_explicit_metadata(repository: Path) -> None:
    (repository / "tracked.txt").write_text("changed line\n" * 100, encoding="utf-8")
    dispatcher = ToolDispatcher(
        ToolCatalog({"git_diff": GitDiffTool(repository, max_output_bytes=160)})
    )

    result = await dispatcher.execute(ToolUseBlock("diff", "git_diff", {}))

    assert result.is_error is False
    assert result.metadata["truncated"] is True
    assert "output truncated" in result.content.lower()
    assert len(result.content.encode("utf-8")) <= 220


@pytest.mark.asyncio
async def test_builtin_catalog_exposes_both_read_only_git_tools(repository: Path) -> None:
    tools = await BuiltinToolSource(repository).list_tools()

    names = {tool.name for tool in tools}
    assert {"git_status", "git_diff"} <= names
