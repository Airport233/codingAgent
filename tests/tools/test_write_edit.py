from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from coding_agent.tools.files import (
    EditFileInput,
    EditFileTool,
    MkdirInput,
    MkdirTool,
    WriteFileInput,
    WriteFileTool,
)

from coding_agent.tools.base import RecoverableToolError
from coding_agent.tools.builtin import ReadFileInput, ReadFileTool
from coding_agent.tools.workspace import AtomicFileWriter, ReadSet, WorkspaceGuard


@pytest.fixture
def workspace() -> Path:
    path = Path.cwd() / ".tmp" / "write-edit"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.asyncio
async def test_write_file_creates_content_empty_files_and_parent_directories(
    workspace: Path,
) -> None:
    tool = WriteFileTool(workspace, ReadSet())

    result = await tool.execute(
        WriteFileInput(path="nested/content.txt", content="hello\n", create_parents=True)
    )
    await tool.execute(WriteFileInput(path="empty.txt"))

    assert (workspace / "nested/content.txt").read_bytes() == b"hello\n"
    assert (workspace / "empty.txt").read_bytes() == b""
    assert "created" in result.content.lower()


@pytest.mark.asyncio
async def test_write_file_requires_explicit_parent_and_overwrite_intent(
    workspace: Path,
) -> None:
    existing = workspace / "existing.txt"
    existing.write_text("old", encoding="utf-8")
    tool = WriteFileTool(workspace, ReadSet())

    with pytest.raises(RecoverableToolError, match="parent"):
        await tool.execute(WriteFileInput(path="missing/file.txt", content="new"))
    with pytest.raises(RecoverableToolError, match="overwrite"):
        await tool.execute(WriteFileInput(path="existing.txt", content="new"))

    assert existing.read_text(encoding="utf-8") == "old"


@pytest.mark.asyncio
async def test_overwrite_requires_an_unchanged_read_set_entry(workspace: Path) -> None:
    target = workspace / "existing.txt"
    target.write_text("old", encoding="utf-8")
    read_set = ReadSet()
    reader = ReadFileTool(workspace, read_set=read_set)
    writer = WriteFileTool(workspace, read_set)

    with pytest.raises(RecoverableToolError, match="read"):
        await writer.execute(WriteFileInput(path="existing.txt", content="new", overwrite=True))

    await reader.execute(ReadFileInput(path="existing.txt"))
    target.write_text("changed elsewhere", encoding="utf-8")

    with pytest.raises(RecoverableToolError, match="changed"):
        await writer.execute(WriteFileInput(path="existing.txt", content="new", overwrite=True))

    assert target.read_text(encoding="utf-8") == "changed elsewhere"


@pytest.mark.asyncio
async def test_edit_replaces_inclusive_lines_and_preserves_crlf(workspace: Path) -> None:
    target = workspace / "sample.txt"
    target.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    read_set = ReadSet()
    reader = ReadFileTool(workspace, read_set=read_set)
    editor = EditFileTool(workspace, read_set)
    await reader.execute(ReadFileInput(path="sample.txt", start_line=2, end_line=3))

    result = await editor.execute(
        EditFileInput(
            path="sample.txt",
            start_line=2,
            end_line=3,
            expected_content="two\nthree",
            replacement="second\nthird",
        )
    )

    assert target.read_bytes() == b"one\r\nsecond\r\nthird\r\n"
    assert "lines 2-3" in result.content


@pytest.mark.asyncio
async def test_edit_rejects_unread_ranges_and_expected_content_mismatch(
    workspace: Path,
) -> None:
    target = workspace / "sample.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")
    read_set = ReadSet()
    reader = ReadFileTool(workspace, read_set=read_set)
    editor = EditFileTool(workspace, read_set)
    await reader.execute(ReadFileInput(path="sample.txt", start_line=2, end_line=2))

    with pytest.raises(RecoverableToolError, match="not read"):
        await editor.execute(
            EditFileInput(
                path="sample.txt",
                start_line=1,
                end_line=2,
                expected_content="one\ntwo",
                replacement="changed",
            )
        )
    with pytest.raises(RecoverableToolError, match="expected_content"):
        await editor.execute(
            EditFileInput(
                path="sample.txt",
                start_line=2,
                end_line=2,
                expected_content="wrong",
                replacement="changed",
            )
        )

    assert target.read_text(encoding="utf-8") == "one\ntwo\nthree\n"


@pytest.mark.asyncio
async def test_mkdir_is_recursive_idempotent_and_rejects_protected_paths(
    workspace: Path,
) -> None:
    tool = MkdirTool(workspace)

    await tool.execute(MkdirInput(path="a/b/c"))
    result = await tool.execute(MkdirInput(path="a/b/c"))

    assert (workspace / "a/b/c").is_dir()
    assert "already exists" in result.content.lower()
    with pytest.raises(RecoverableToolError, match="protected"):
        await tool.execute(MkdirInput(path=".git/hooks"))


def test_workspace_guard_rejects_escape_and_atomic_failure_preserves_original(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = WorkspaceGuard(workspace)
    target = workspace / "target.txt"
    target.write_text("original", encoding="utf-8")

    with pytest.raises(RecoverableToolError, match="outside"):
        guard.resolve_for_write("../outside.txt")

    def fail_replace(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(RecoverableToolError, match="atomically"):
        AtomicFileWriter().write(target, b"replacement")

    assert target.read_text(encoding="utf-8") == "original"
    assert not list(workspace.glob(".coding-agent-*.tmp"))
