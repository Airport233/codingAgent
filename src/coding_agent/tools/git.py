from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from coding_agent.tools.base import RecoverableToolError, ToolOutput
from coding_agent.tools.workspace import WorkspaceGuard


class GitStatusInput(BaseModel):
    include_untracked: bool = True


class GitDiffInput(BaseModel):
    scope: Literal["unstaged", "staged"] = "unstaged"
    path: str | None = Field(default=None, min_length=1, max_length=1_000)
    context_lines: int = Field(default=3, ge=0, le=20)


class _GitRunner:
    def __init__(
        self,
        workspace: Path,
        *,
        git_executable: str | None = None,
        timeout_seconds: float = 15,
        max_output_bytes: int = 65_536,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self.workspace = workspace.resolve()
        self.guard = WorkspaceGuard(self.workspace)
        self.executable = git_executable or shutil.which("git")
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes

    async def run(self, arguments: tuple[str, ...]) -> tuple[str, dict[str, object]]:
        if self.executable is None:
            raise RecoverableToolError("Git executable is unavailable")
        environment = dict(os.environ)
        environment.update(
            {
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_PAGER": "cat",
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
        try:
            process = await asyncio.create_subprocess_exec(
                self.executable,
                "-c",
                "color.ui=false",
                "-c",
                "core.fsmonitor=false",
                "--no-pager",
                "-C",
                str(self.workspace),
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=environment,
            )
        except OSError as error:
            raise RecoverableToolError("Unable to start Git") from error
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout_seconds
            )
        except TimeoutError as error:
            process.kill()
            await process.wait()
            raise RecoverableToolError("Git command timed out") from error
        if process.returncode != 0:
            detail, _truncated = _bounded_text(stderr, min(self.max_output_bytes, 4_096))
            raise RecoverableToolError(detail or f"Git exited with code {process.returncode}")
        content, truncated = _bounded_text(stdout, self.max_output_bytes)
        return content, {
            "exit_code": process.returncode,
            "raw_output_bytes": len(stdout),
            "truncated": truncated,
        }


class GitStatusTool:
    name = "git_status"
    description = (
        "Inspect the current Git branch and working tree changes without modifying the repository."
    )
    input_model = GitStatusInput

    def __init__(
        self,
        workspace: Path,
        *,
        git_executable: str | None = None,
        timeout_seconds: float = 15,
        max_output_bytes: int = 65_536,
    ) -> None:
        self._runner = _GitRunner(
            workspace,
            git_executable=git_executable,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    async def execute(self, arguments: BaseModel) -> ToolOutput:
        parsed = GitStatusInput.model_validate(arguments)
        untracked = "all" if parsed.include_untracked else "no"
        content, metadata = await self._runner.run(
            ("status", "--short", "--branch", f"--untracked-files={untracked}")
        )
        return ToolOutput(content or "Working tree clean", metadata)


class GitDiffTool:
    name = "git_diff"
    description = (
        "Inspect unstaged or staged Git changes, optionally limited to one workspace path. "
        "This tool is read-only."
    )
    input_model = GitDiffInput

    def __init__(
        self,
        workspace: Path,
        *,
        git_executable: str | None = None,
        timeout_seconds: float = 15,
        max_output_bytes: int = 65_536,
    ) -> None:
        self._runner = _GitRunner(
            workspace,
            git_executable=git_executable,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    async def execute(self, arguments: BaseModel) -> ToolOutput:
        parsed = GitDiffInput.model_validate(arguments)
        command = [
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            f"--unified={parsed.context_lines}",
        ]
        if parsed.scope == "staged":
            command.append("--cached")
        if parsed.path is not None:
            resolved = self._runner.guard.resolve(parsed.path)
            relative = resolved.relative_to(self._runner.workspace).as_posix()
            command.extend(("--", relative))
        content, metadata = await self._runner.run(tuple(command))
        metadata["scope"] = parsed.scope
        if parsed.path is not None:
            metadata["path"] = parsed.path
        return ToolOutput(content or "No matching changes", metadata)


def _bounded_text(raw: bytes, limit: int) -> tuple[str, bool]:
    if len(raw) <= limit:
        return raw.decode("utf-8", errors="replace").strip(), False
    selected = raw[:limit].decode("utf-8", errors="ignore").rstrip()
    return f"{selected}\n[output truncated at {limit} bytes]", True
