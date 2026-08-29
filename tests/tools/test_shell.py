from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from coding_agent.tools.base import RecoverableToolError
from coding_agent.tools.shell import (
    PosixShellBackend,
    ShellConfig,
    ShellInput,
    ShellPolicy,
    ShellTool,
    WindowsPowerShellBackend,
    classify_shell_command,
    create_shell_backend,
)
from coding_agent.tools.workspace import WorkspaceGuard


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


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"default_timeout_seconds": 0}, "positive"),
        (
            {"default_timeout_seconds": 2, "max_timeout_seconds": 1},
            "at least the default",
        ),
        ({"max_output_bytes": 0}, "positive"),
        ({"termination_grace_seconds": -1}, "cannot be negative"),
    ],
)
def test_shell_config_rejects_unsafe_bounds(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ShellConfig(**overrides)


def test_shell_policy_marks_risky_commands_without_executing_policy_decisions(
    workspace: Path,
) -> None:
    policy = ShellPolicy(
        workspace,
        ShellConfig(),
        source_environment={"PATH": "safe", "LC_ALL": "C", "SECRET": "hidden"},
        platform_name="darwin",
    )

    request = policy.prepare(ShellInput(command="git reset --hard"))

    assert request.risk_level == "elevated"
    assert request.environment == {"PATH": "safe", "LC_ALL": "C"}
    with pytest.raises(RecoverableToolError, match="directory"):
        policy.prepare(ShellInput(command="pwd", cwd="missing"))


@pytest.mark.asyncio
async def test_shell_start_failures_are_recoverable(workspace: Path) -> None:
    class FailingBackend:
        async def start(self, command: str, *, cwd: Path, environment: dict[str, str]):
            raise FileNotFoundError("missing shell")

        async def terminate(self, process, grace_seconds: float) -> None:
            raise AssertionError("no process should have started")

    tool = ShellTool(workspace, backend=FailingBackend())

    with pytest.raises(RecoverableToolError, match="configured shell"):
        await tool.execute(ShellInput(command="echo never-runs"))


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

    result = await _execute_or_skip_sandbox(tool, ShellInput(command=command, cwd="nested"))

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
    if sys.platform == "win32":
        command = "Write-Output ('x' * 200)"
    else:
        command = f'"{sys.executable}" -c "print(\'x\'*200)"'

    result = await _execute_or_skip_sandbox(tool, ShellInput(command=command))

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

    result = await _execute_or_skip_sandbox(tool, ShellInput(command=command))

    assert result.metadata["timed_out"] is True
    assert "timed_out: true" in result.content


@pytest.mark.asyncio
async def test_shell_cancellation_terminates_before_propagating(workspace: Path) -> None:
    tool = ShellTool(
        workspace,
        ShellConfig(default_timeout_seconds=10, termination_grace_seconds=0.1),
    )
    command = f'"{sys.executable}" -c "import time; time.sleep(5)"'
    await _execute_or_skip_sandbox(tool, ShellInput(command="echo shell-probe"))
    task = asyncio.create_task(tool.execute(ShellInput(command=command)))
    await asyncio.sleep(0.2)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)


def test_classify_shell_command_allows_safe_commands_to_follow_the_configured_mode(
    workspace: Path,
) -> None:
    guard = WorkspaceGuard(workspace)

    verdict = classify_shell_command("pwd", guard)

    assert verdict.tier == "low"
    assert verdict.forced_action is None
    assert verdict.escapes_workspace is False
    assert verdict.touches_sensitive_path is False


def test_classify_shell_command_forces_ask_for_builtin_elevated_patterns(
    workspace: Path,
) -> None:
    guard = WorkspaceGuard(workspace)

    verdict = classify_shell_command("git push origin main", guard)

    assert verdict.tier == "elevated"
    assert verdict.forced_action == "ask"


def test_classify_shell_command_detects_cd_escaping_the_workspace(workspace: Path) -> None:
    guard = WorkspaceGuard(workspace)

    verdict = classify_shell_command("cd .. && ls", guard)

    assert verdict.escapes_workspace is True
    assert verdict.forced_action == "ask"


def test_classify_shell_command_allows_cd_within_the_workspace(workspace: Path) -> None:
    (workspace / "nested").mkdir()
    guard = WorkspaceGuard(workspace)

    verdict = classify_shell_command("cd nested && ls", guard)

    assert verdict.escapes_workspace is False
    assert verdict.forced_action is None


def test_classify_shell_command_detects_absolute_path_argument_outside_workspace(
    workspace: Path,
) -> None:
    guard = WorkspaceGuard(workspace)

    verdict = classify_shell_command("cat /etc/passwd", guard)

    assert verdict.escapes_workspace is True
    assert verdict.forced_action == "ask"


def test_classify_shell_command_allows_absolute_path_argument_inside_workspace(
    workspace: Path,
) -> None:
    guard = WorkspaceGuard(workspace)
    target = (workspace / "inside.txt").resolve()
    target.write_text("hello", encoding="utf-8")

    verdict = classify_shell_command(f"cat {target}", guard)

    assert verdict.escapes_workspace is False


def test_classify_shell_command_detects_sensitive_paths_outside_file_tools(
    workspace: Path,
) -> None:
    guard = WorkspaceGuard(workspace)

    verdict = classify_shell_command("cat .env", guard)

    assert verdict.touches_sensitive_path is True
    assert verdict.forced_action == "ask"

    git_verdict = classify_shell_command("rm -rf .git", guard)
    assert git_verdict.touches_sensitive_path is True


def test_classify_shell_command_configured_rules_take_precedence(workspace: Path) -> None:
    guard = WorkspaceGuard(workspace)

    denied = classify_shell_command("rm -rf build", guard, {"rm *": "ask"})
    allowed = classify_shell_command("git status", guard, {"git status": "allow"})

    assert denied.forced_action == "ask"
    assert denied.matched_rule == "rm *"
    assert allowed.forced_action == "allow"


def test_classify_shell_command_tolerates_unparseable_quoting(workspace: Path) -> None:
    guard = WorkspaceGuard(workspace)

    verdict = classify_shell_command("echo 'unterminated", guard)

    assert verdict.escapes_workspace is False


def test_classify_shell_command_tolerates_trailing_chain_separator(workspace: Path) -> None:
    guard = WorkspaceGuard(workspace)

    verdict = classify_shell_command("ls &&", guard)

    assert verdict.escapes_workspace is False


def test_classify_shell_command_treats_unresolvable_paths_as_safe(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = WorkspaceGuard(workspace)
    original_resolve = Path.resolve

    def failing_resolve(self: Path, *args: object, **kwargs: object) -> Path:
        if self.name == "bad":
            raise OSError("simulated resolution failure")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", failing_resolve)

    verdict = classify_shell_command("cat nested/bad", guard)

    assert verdict.escapes_workspace is False


async def _execute_or_skip_sandbox(tool: ShellTool, arguments: ShellInput):
    try:
        return await tool.execute(arguments)
    except RecoverableToolError as error:
        cause = error.__cause__
        if sys.platform == "win32" and isinstance(cause, PermissionError):
            pytest.skip("local sandbox blocks Windows asyncio named pipes")
        raise
