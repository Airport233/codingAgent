from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from coding_agent.tools.base import RecoverableToolError, Tool, ToolOutput


class ReadFileInput(BaseModel):
    path: str = Field(min_length=1)


class ReadFileTool:
    name = "read_file"
    description = "Read a UTF-8 text file inside the current workspace."
    input_model = ReadFileInput

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve()

    async def execute(self, arguments: BaseModel) -> ToolOutput:
        parsed = ReadFileInput.model_validate(arguments)
        requested = Path(parsed.path)
        if requested.is_absolute():
            raise RecoverableToolError("Path is outside the workspace")
        resolved = (self._workspace / requested).resolve()
        if not resolved.is_relative_to(self._workspace):
            raise RecoverableToolError("Path is outside the workspace")
        if not resolved.is_file():
            raise RecoverableToolError("File does not exist")
        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise RecoverableToolError("File is not valid UTF-8 text") from error
        numbered = "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(content.splitlines(), start=1)
        )
        return ToolOutput(content=numbered)


class BuiltinToolSource:
    source_id = "builtin"

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    async def list_tools(self) -> tuple[Tool, ...]:
        return (ReadFileTool(self._workspace),)
