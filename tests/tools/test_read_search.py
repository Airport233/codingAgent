from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from coding_agent.tools.base import RecoverableToolError
from coding_agent.tools.builtin import ReadFileInput, ReadFileTool
from coding_agent.tools.search import CodeSearchInput, CodeSearchTool


@pytest.fixture
def workspace() -> Path:
    path = Path.cwd() / ".tmp" / "read-search"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.asyncio
async def test_read_file_supports_inclusive_line_ranges(workspace: Path) -> None:
    (workspace / "sample.py").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    tool = ReadFileTool(workspace)

    result = await tool.execute(ReadFileInput(path="sample.py", start_line=2, end_line=3))

    assert result.content == "2: two\n3: three"


@pytest.mark.asyncio
async def test_read_file_rejects_binary_and_oversized_content(workspace: Path) -> None:
    (workspace / "binary.bin").write_bytes(b"text\x00binary")
    (workspace / "large.txt").write_text("123456789", encoding="utf-8")

    with pytest.raises(RecoverableToolError, match="binary"):
        await ReadFileTool(workspace).execute(ReadFileInput(path="binary.bin"))
    with pytest.raises(RecoverableToolError, match="too large"):
        await ReadFileTool(workspace, max_bytes=8).execute(ReadFileInput(path="large.txt"))


@pytest.mark.asyncio
async def test_code_search_fallback_filters_glob_case_and_limit(workspace: Path) -> None:
    (workspace / "a.py").write_text("Needle first\nneedle second\n", encoding="utf-8")
    (workspace / "b.py").write_text("NEEDLE third\n", encoding="utf-8")
    (workspace / "notes.txt").write_text("needle ignored\n", encoding="utf-8")
    tool = CodeSearchTool(workspace, rg_executable=None)

    result = await tool.execute(
        CodeSearchInput(
            query="needle",
            glob="*.py",
            case_sensitive=False,
            regex=False,
            max_results=2,
        )
    )

    assert result.content.splitlines() == [
        "a.py:1:Needle first",
        "a.py:2:needle second",
        "[results truncated at 2 matches]",
    ]


@pytest.mark.asyncio
async def test_code_search_supports_regular_expressions(workspace: Path) -> None:
    (workspace / "main.py").write_text("value_1\nvalue_x\nvalue_22\n", encoding="utf-8")
    tool = CodeSearchTool(workspace, rg_executable=None)

    result = await tool.execute(CodeSearchInput(query=r"value_\d+$", regex=True, max_results=10))

    assert result.content.splitlines() == ["main.py:1:value_1", "main.py:3:value_22"]


@pytest.mark.asyncio
async def test_code_search_invalid_regex_is_recoverable(workspace: Path) -> None:
    tool = CodeSearchTool(workspace, rg_executable=None)

    with pytest.raises(RecoverableToolError, match="Invalid regular expression"):
        await tool.execute(CodeSearchInput(query="[", regex=True))


def test_ripgrep_json_parser_ignores_non_matches_and_normalizes_paths() -> None:
    output = b"\n".join(
        (
            b'{"type":"begin","data":{"path":{"text":"src\\\\a.py"}}}',
            b'{"type":"match","data":{"path":{"text":"src\\\\a.py"},'
            b'"lines":{"text":"needle\\n"},"line_number":4}}',
        )
    )

    assert CodeSearchTool._parse_ripgrep_json(output, max_results=10) == ["src/a.py:4:needle"]


@pytest.mark.asyncio
async def test_ripgrep_backend_is_used_when_available(workspace: Path) -> None:
    rg = shutil.which("rg")
    if rg is None:
        pytest.skip("ripgrep is not installed on this platform")
    (workspace / "main.py").write_text("needle here\n", encoding="utf-8")
    tool = CodeSearchTool(workspace, rg_executable=rg)

    result = await tool.execute(CodeSearchInput(query="needle", glob="*.py"))

    assert "main.py:1:needle here" in result.content
