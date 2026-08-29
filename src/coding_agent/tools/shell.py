from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, Field

from coding_agent.approval import ApprovalMode
from coding_agent.tools.base import RecoverableToolError, ToolOutput
from coding_agent.tools.workspace import WorkspaceGuard

RiskLevel = Literal["low", "elevated"]


@dataclass(frozen=True, slots=True)
class ShellConfig:
    mode: ApprovalMode = "auto"
    default_timeout_seconds: float = 30
    max_timeout_seconds: float = 300
    max_output_bytes: int = 65_536
    termination_grace_seconds: float = 1
    executable: str | None = None

    def __post_init__(self) -> None:
        if self.default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")
        if self.max_timeout_seconds < self.default_timeout_seconds:
            raise ValueError("max_timeout_seconds must be at least the default")
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        if self.termination_grace_seconds < 0:
            raise ValueError("termination_grace_seconds cannot be negative")


class ShellInput(BaseModel):
    command: str = Field(min_length=1)
    cwd: str = "."
    timeout_seconds: float | None = Field(default=None, gt=0)


@dataclass(frozen=True, slots=True)
class ShellRequest:
    command: str
    cwd: Path
    timeout_seconds: float
    environment: dict[str, str]
    approval_mode: ApprovalMode
    risk_level: RiskLevel


class ShellPolicy:
    _WINDOWS_ENVIRONMENT = frozenset(
        {
            "comspec",
            "homedrive",
            "homepath",
            "localappdata",
            "path",
            "pathext",
            "systemroot",
            "temp",
            "tmp",
            "userprofile",
            "windir",
        }
    )
    _POSIX_ENVIRONMENT = frozenset({"home", "lang", "path", "term", "tmpdir"})
    _ELEVATED_PATTERNS = (
        re.compile(r"(^|\s)(rm|rmdir|del|erase|format)(\s|$)", re.IGNORECASE),
        re.compile(r"git\s+(push|reset|clean)\b", re.IGNORECASE),
        re.compile(r"\b(sudo|runas)\b", re.IGNORECASE),
    )

    def __init__(
        self,
        workspace: Path,
        config: ShellConfig,
        *,
        source_environment: dict[str, str] | None = None,
        platform_name: str | None = None,
    ) -> None:
        self._guard = WorkspaceGuard(workspace)
        self._config = config
        self._source_environment = source_environment or dict(os.environ)
        self._platform_name = platform_name or sys.platform

    def prepare(self, arguments: ShellInput) -> ShellRequest:
        cwd = self._guard.resolve(arguments.cwd)
        if not cwd.is_dir():
            raise RecoverableToolError("Shell cwd must be a directory inside the workspace")
        timeout = arguments.timeout_seconds or self._config.default_timeout_seconds
        if timeout > self._config.max_timeout_seconds:
            raise RecoverableToolError(
                f"timeout exceeds maximum of {self._config.max_timeout_seconds:g} seconds"
            )
        return ShellRequest(
            command=arguments.command,
            cwd=cwd,
            timeout_seconds=timeout,
            environment=self._safe_environment(),
            approval_mode=self._config.mode,
            risk_level=self._risk_level(arguments.command),
        )

    def _safe_environment(self) -> dict[str, str]:
        windows = self._platform_name == "win32"
        allowed = self._WINDOWS_ENVIRONMENT if windows else self._POSIX_ENVIRONMENT
        result: dict[str, str] = {}
        for key, value in self._source_environment.items():
            folded = key.casefold()
            if folded in allowed or (not windows and folded.startswith("lc_")):
                result[key] = value
        return result

    def _risk_level(self, command: str) -> RiskLevel:
        if any(pattern.search(command) for pattern in self._ELEVATED_PATTERNS):
            return "elevated"
        return "low"


class ShellBackend(Protocol):
    async def start(
        self, command: str, *, cwd: Path, environment: dict[str, str]
    ) -> asyncio.subprocess.Process: ...

    async def terminate(
        self, process: asyncio.subprocess.Process, grace_seconds: float
    ) -> None: ...


class WindowsPowerShellBackend:
    def __init__(self, executable: str | None = None) -> None:
        self.executable = (
            executable or shutil.which("pwsh") or shutil.which("powershell") or "powershell.exe"
        )

    def command_args(self, command: str) -> tuple[str, ...]:
        return (
            self.executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        )

    async def start(
        self, command: str, *, cwd: Path, environment: dict[str, str]
    ) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            *self.command_args(command),
            cwd=cwd,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=cast(Any, subprocess).CREATE_NEW_PROCESS_GROUP,
        )

    async def terminate(self, process: asyncio.subprocess.Process, grace_seconds: float) -> None:
        if process.returncode is not None:
            return
        try:
            process.send_signal(cast(Any, signal).CTRL_BREAK_EVENT)
            await asyncio.wait_for(process.wait(), timeout=grace_seconds)
            return
        except (TimeoutError, OSError, ProcessLookupError):
            pass
        taskkill = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(process.pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await taskkill.wait()
        if process.returncode is None:
            process.kill()
        await process.wait()


class PosixShellBackend:
    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable or _default_posix_shell()

    def command_args(self, command: str) -> tuple[str, ...]:
        return (self.executable, "-lc", command)

    async def start(
        self, command: str, *, cwd: Path, environment: dict[str, str]
    ) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            *self.command_args(command),
            cwd=cwd,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

    async def terminate(self, process: asyncio.subprocess.Process, grace_seconds: float) -> None:
        if process.returncode is not None:
            return
        try:
            _kill_process_group(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=grace_seconds)
            return
        except TimeoutError:
            pass
        with contextlib.suppress(ProcessLookupError):
            _kill_process_group(process.pid, cast(Any, signal).SIGKILL)
        await process.wait()


def create_shell_backend(
    *, platform_name: str | None = None, executable: str | None = None
) -> ShellBackend:
    if (platform_name or sys.platform) == "win32":
        return WindowsPowerShellBackend(executable)
    return PosixShellBackend(executable)


class _BoundedOutputCollector:
    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._stored_bytes = 0
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.stdout_bytes = 0
        self.stderr_bytes = 0

    async def read(self, stream: asyncio.StreamReader, *, stderr: bool) -> None:
        destination = self.stderr if stderr else self.stdout
        while chunk := await stream.read(8192):
            if stderr:
                self.stderr_bytes += len(chunk)
            else:
                self.stdout_bytes += len(chunk)
            remaining = self._max_bytes - self._stored_bytes
            if remaining > 0:
                selected = chunk[:remaining]
                destination.extend(selected)
                self._stored_bytes += len(selected)

    @property
    def truncated(self) -> bool:
        return self.stdout_bytes + self.stderr_bytes > self._stored_bytes


class ShellTool:
    name = "shell"
    description = "Run a command in the workspace shell with bounded output and timeout."
    input_model = ShellInput

    def __init__(
        self,
        workspace: Path,
        config: ShellConfig | None = None,
        *,
        backend: ShellBackend | None = None,
    ) -> None:
        self._config = config or ShellConfig()
        self._policy = ShellPolicy(workspace, self._config)
        self._backend = backend or create_shell_backend(executable=self._config.executable)

    async def execute(self, arguments: BaseModel) -> ToolOutput:
        parsed = ShellInput.model_validate(arguments)
        request = self._policy.prepare(parsed)
        started = time.monotonic()
        try:
            process = await self._backend.start(
                request.command, cwd=request.cwd, environment=request.environment
            )
        except OSError as error:
            raise RecoverableToolError("Unable to start the configured shell") from error
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Shell backend did not create output pipes")
        collector = _BoundedOutputCollector(self._config.max_output_bytes)
        readers = (
            asyncio.create_task(collector.read(process.stdout, stderr=False)),
            asyncio.create_task(collector.read(process.stderr, stderr=True)),
        )
        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=request.timeout_seconds)
        except TimeoutError:
            timed_out = True
            await self._backend.terminate(process, self._config.termination_grace_seconds)
        except asyncio.CancelledError:
            await self._backend.terminate(process, self._config.termination_grace_seconds)
            await asyncio.gather(*readers)
            raise
        await asyncio.gather(*readers)
        duration_ms = round((time.monotonic() - started) * 1000)
        metadata: dict[str, object] = {
            "exit_code": process.returncode,
            "duration_ms": duration_ms,
            "timed_out": timed_out,
            "cancelled": False,
            "truncated": collector.truncated,
            "stdout_bytes": collector.stdout_bytes,
            "stderr_bytes": collector.stderr_bytes,
            "approval_mode": request.approval_mode,
            "risk_level": request.risk_level,
        }
        content = _format_result(collector, metadata)
        return ToolOutput(content=content, metadata=metadata)


def _default_posix_shell() -> str:
    configured = os.environ.get("SHELL")
    if configured and Path(configured).is_file():
        return configured
    for candidate in ("/bin/zsh", "/bin/bash", "/bin/sh"):
        if Path(candidate).is_file():
            return candidate
    return "/bin/sh"


def _kill_process_group(process_id: int, requested_signal: int) -> None:
    killpg = cast(Any, os).killpg
    killpg(process_id, requested_signal)


def _format_result(collector: _BoundedOutputCollector, metadata: dict[str, object]) -> str:
    stdout = collector.stdout.decode("utf-8", errors="replace").rstrip()
    stderr = collector.stderr.decode("utf-8", errors="replace").rstrip()
    lines = [
        f"exit_code: {metadata['exit_code']}",
        f"duration_ms: {metadata['duration_ms']}",
        f"timed_out: {str(metadata['timed_out']).lower()}",
        "stdout:",
        stdout,
        "stderr:",
        stderr,
    ]
    if collector.truncated:
        lines.append(
            "[output truncated; "
            f"stdout={collector.stdout_bytes} bytes, stderr={collector.stderr_bytes} bytes]"
        )
    return "\n".join(lines)
