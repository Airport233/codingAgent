from __future__ import annotations

import shutil
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from coding_agent.tools.base import RecoverableToolError, Tool, ToolOutput
from coding_agent.tools.search import CodeSearchTool
from coding_agent.tools.workspace import WorkspaceGuard


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

    def __init__(self, workspace: Path, max_bytes: int = 1_048_576) -> None:
        self._guard = WorkspaceGuard(workspace)
        self._max_bytes = max_bytes

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
        numbered = "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(selected, start=parsed.start_line)
        )
        return ToolOutput(content=numbered)


class BuiltinToolSource:
    source_id = "builtin"

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    async def list_tools(self) -> tuple[Tool, ...]:
        return (
            ReadFileTool(self._workspace),
            CodeSearchTool(self._workspace, rg_executable=shutil.which("rg")),
        )
