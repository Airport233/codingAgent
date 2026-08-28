from __future__ import annotations

import shutil
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from coding_agent.plan import PlanState
from coding_agent.tools.base import RecoverableToolError, Tool, ToolOutput
from coding_agent.tools.files import EditFileTool, MkdirTool, WriteFileTool
from coding_agent.tools.git import GitDiffTool, GitStatusTool
from coding_agent.tools.plan import UpdatePlanTool
from coding_agent.tools.search import CodeSearchTool
from coding_agent.tools.shell import ShellConfig, ShellTool
from coding_agent.tools.workspace import ReadSet, WorkspaceGuard


class ReadFileInput(BaseModel):
    path: str = Field(min_length=1)
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> ReadFileInput:
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class ReadFileTool:
    name = "read_file"
    description = "Read a UTF-8 text file inside the current workspace."
    input_model = ReadFileInput

    def __init__(
        self,
        workspace: Path,
        max_bytes: int = 1_048_576,
        read_set: ReadSet | None = None,
    ) -> None:
        self._guard = WorkspaceGuard(workspace)
        self._max_bytes = max_bytes
        self._read_set = read_set

    async def execute(self, arguments: BaseModel) -> ToolOutput:
        parsed = ReadFileInput.model_validate(arguments)
        resolved = self._guard.resolve(parsed.path)
        if not resolved.is_file():
            raise RecoverableToolError("File does not exist")
        if resolved.stat().st_size > self._max_bytes:
            raise RecoverableToolError("File is too large to read")
        raw = resolved.read_bytes()
        if b"\x00" in raw:
            raise RecoverableToolError("File appears to be binary")
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RecoverableToolError("File is not valid UTF-8 text") from error
        lines = content.splitlines()
        if lines and parsed.start_line > len(lines):
            raise RecoverableToolError("start_line is beyond the end of the file")
        end_line = parsed.end_line or len(lines)
        selected = lines[parsed.start_line - 1 : end_line]
        if self._read_set is not None:
            self._read_set.record(
                resolved,
                raw,
                start_line=parsed.start_line,
                end_line=end_line,
                mtime_ns=resolved.stat().st_mtime_ns,
            )
        numbered = "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(selected, start=parsed.start_line)
        )
        return ToolOutput(content=numbered or "(file is empty)")


class BuiltinToolSource:
    source_id = "builtin"

    def __init__(
        self,
        workspace: Path,
        *,
        shell_config: ShellConfig | None = None,
        plan_state: PlanState | None = None,
    ) -> None:
        self._workspace = workspace
        self._read_set = ReadSet()
        self._shell_config = shell_config
        self._plan_state = plan_state or PlanState()

    async def list_tools(self) -> tuple[Tool, ...]:
        return (
            ReadFileTool(self._workspace, read_set=self._read_set),
            CodeSearchTool(self._workspace, rg_executable=shutil.which("rg")),
            WriteFileTool(self._workspace, self._read_set),
            MkdirTool(self._workspace),
            EditFileTool(self._workspace, self._read_set),
            ShellTool(self._workspace, self._shell_config),
            UpdatePlanTool(self._plan_state),
            GitStatusTool(self._workspace),
            GitDiffTool(self._workspace),
        )
