from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

import pytest
from coding_agent.tools.shell import (
    PosixShellBackend,
    ShellConfig,
    ShellInput,
    ShellPolicy,
    ShellTool,
    WindowsPowerShellBackend,
    create_shell_backend,
)

from coding_agent.tools.base import RecoverableToolError


@pytest.fixture
def workspace() -> Path:
    path = Path.cwd() / ".tmp" / "shell"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_backend_factory_keeps_platform_command_semantics_separate() -> None:
    windows = create_shell_backend(platform_name="win32", executable="pwsh.exe")
    posix = create_shell_backend(platform_name="darwin", executable="/bin/zsh")

    assert isinstance(windows, WindowsPowerShellBackend)
    assert windows.command_args("Write-Output hi") == (
        "pwsh.exe",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Write-Output hi",
    )
    assert isinstance(posix, PosixShellBackend)
    assert posix.command_args("printf hi") == ("/bin/zsh", "-lc", "printf hi")


def test_shell_policy_bounds_cwd_timeout_environment_and_exposes_approval_metadata(
    workspace: Path,
) -> None:
    (workspace / "src").mkdir()
    policy = ShellPolicy(
        workspace,
        ShellConfig(default_timeout_seconds=2, max_timeout_seconds=5),
        source_environment={
            "PATH": os.environ.get("PATH", ""),
            "HOME": "safe-home",
            "ANTHROPIC_API_KEY": "must-not-leak",
            "INTERNAL_TOKEN": "must-not-leak",
        },
        platform_name="darwin",
    )

    request = policy.prepare(ShellInput(command="git status", cwd="src", timeout_seconds=4))

    assert request.cwd == (workspace / "src").resolve()
    assert request.timeout_seconds == 4
    assert request.environment["HOME"] == "safe-home"
    assert "ANTHROPIC_API_KEY" not in request.environment
    assert "INTERNAL_TOKEN" not in request.environment
    assert request.approval_mode == "auto"
    assert request.risk_level == "low"

    with pytest.raises(RecoverableToolError, match="workspace"):
        policy.prepare(ShellInput(command="pwd", cwd="../outside"))
    with pytest.raises(RecoverableToolError, match="maximum"):
        policy.prepare(ShellInput(command="pwd", timeout_seconds=6))


@pytest.mark.asyncio
async def test_shell_runs_in_requested_directory_and_reports_exit_status(
    workspace: Path,
) -> None:
    (workspace / "nested").mkdir()
    tool = ShellTool(workspace, ShellConfig(max_output_bytes=4096))
    if sys.platform == "win32":
        command = "Write-Output shell-ok; [Environment]::CurrentDirectory; exit 7"
    else:
        command = "printf 'shell-ok\\n'; pwd; exit 7"

    result = await tool.execute(ShellInput(command=command, cwd="nested"))

    assert "shell-ok" in result.content
    assert "exit_code: 7" in result.content
    assert result.metadata["exit_code"] == 7
    assert result.metadata["timed_out"] is False
    assert result.metadata["approval_mode"] == "auto"


@pytest.mark.asyncio
async def test_shell_streams_with_a_bounded_output_and_marks_truncation(
    workspace: Path,
) -> None:
    tool = ShellTool(workspace, ShellConfig(max_output_bytes=64))
    command = f'"{sys.executable}" -c "print(\'x\'*200)"'

    result = await tool.execute(ShellInput(command=command))

    assert result.metadata["truncated"] is True
    assert result.metadata["stdout_bytes"] >= 200
    assert "output truncated" in result.content
    assert len(result.content) < 512


@pytest.mark.asyncio
async def test_shell_timeout_terminates_the_process_group(workspace: Path) -> None:
    tool = ShellTool(
        workspace,
        ShellConfig(
            default_timeout_seconds=0.1,
            max_timeout_seconds=1,
            termination_grace_seconds=0.1,
        ),
    )
    command = f'"{sys.executable}" -c "import time; time.sleep(5)"'

    result = await tool.execute(ShellInput(command=command))

    assert result.metadata["timed_out"] is True
    assert "timed_out: true" in result.content


@pytest.mark.asyncio
async def test_shell_cancellation_terminates_before_propagating(workspace: Path) -> None:
    tool = ShellTool(
        workspace,
        ShellConfig(default_timeout_seconds=10, termination_grace_seconds=0.1),
    )
    command = f'"{sys.executable}" -c "import time; time.sleep(5)"'
    task = asyncio.create_task(tool.execute(ShellInput(command=command)))
    await asyncio.sleep(0.2)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)
