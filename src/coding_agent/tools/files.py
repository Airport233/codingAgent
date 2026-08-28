from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from coding_agent.tools.base import RecoverableToolError, ToolOutput
from coding_agent.tools.workspace import AtomicFileWriter, ReadSet, WorkspaceGuard


class WriteFileInput(BaseModel):
    path: str = Field(min_length=1)
    content: str = ""
    create_parents: bool = False
    overwrite: bool = False
    expected_file_hash: str | None = None


class WriteFileTool:
    name = "write_file"
    description = "Create a UTF-8 file, optionally creating parent directories."
    input_model = WriteFileInput

    def __init__(self, workspace: Path, read_set: ReadSet) -> None:
        self._guard = WorkspaceGuard(workspace)
        self._read_set = read_set
        self._writer = AtomicFileWriter()

    async def execute(self, arguments: BaseModel) -> ToolOutput:
        parsed = WriteFileInput.model_validate(arguments)
        target = self._guard.resolve_for_write(parsed.path)
        existed = target.exists()
        if existed and not target.is_file():
            raise RecoverableToolError("Target exists and is not a file")
        if existed and not parsed.overwrite:
            raise RecoverableToolError("Existing files require overwrite=true")
        if not target.parent.is_dir():
            if not parsed.create_parents:
                raise RecoverableToolError("parent directory does not exist")
            self._guard.resolve_for_write(str(target.parent.relative_to(self._guard.root)))
            target.parent.mkdir(parents=True, exist_ok=True)

        if existed:
            self._read_set.require_current(target, expected_hash=parsed.expected_file_hash)

        raw = parsed.content.encode("utf-8")

        def validate() -> None:
            if existed:
                self._read_set.require_current(target, expected_hash=parsed.expected_file_hash)
            elif target.exists():
                raise RecoverableToolError("Target appeared during write; refusing to overwrite")

        self._writer.write(target, raw, validate_before_replace=validate)
        self._read_set.refresh_after_write(target, raw)
        action = "overwrote" if existed else "created"
        return ToolOutput(content=f"{action.capitalize()} {parsed.path} ({len(raw)} bytes)")


class MkdirInput(BaseModel):
    path: str = Field(min_length=1)


class MkdirTool:
    name = "mkdir"
    description = "Recursively create a directory inside the current workspace."
    input_model = MkdirInput

    def __init__(self, workspace: Path) -> None:
        self._guard = WorkspaceGuard(workspace)

    async def execute(self, arguments: BaseModel) -> ToolOutput:
        parsed = MkdirInput.model_validate(arguments)
        target = self._guard.resolve_for_write(parsed.path)
        if target.exists():
            if not target.is_dir():
                raise RecoverableToolError("Target exists and is not a directory")
            return ToolOutput(content=f"Directory {parsed.path} already exists")
        target.mkdir(parents=True)
        return ToolOutput(content=f"Created directory {parsed.path}")


class EditFileInput(BaseModel):
    path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    expected_content: str
    replacement: str
    expected_file_hash: str | None = None

    @model_validator(mode="after")
    def validate_range(self) -> EditFileInput:
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class EditFileTool:
    name = "edit_file"
    description = "Replace an inclusive, previously read line range in a UTF-8 file."
    input_model = EditFileInput

    def __init__(self, workspace: Path, read_set: ReadSet) -> None:
        self._guard = WorkspaceGuard(workspace)
        self._read_set = read_set
        self._writer = AtomicFileWriter()

    async def execute(self, arguments: BaseModel) -> ToolOutput:
        parsed = EditFileInput.model_validate(arguments)
        target = self._guard.resolve_for_write(parsed.path)
        if not target.is_file():
            raise RecoverableToolError("File does not exist")
        self._read_set.require_current(
            target,
            start_line=parsed.start_line,
            end_line=parsed.end_line,
            expected_hash=parsed.expected_file_hash,
        )
        raw = target.read_bytes()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RecoverableToolError("File is not valid UTF-8 text") from error
        lines = content.splitlines()
        if parsed.end_line > len(lines):
            raise RecoverableToolError("end_line is beyond the end of the file")
        selected = "\n".join(lines[parsed.start_line - 1 : parsed.end_line])
        if selected != _normalize_newlines(parsed.expected_content):
            raise RecoverableToolError("expected_content does not match the selected lines")

        replacement_lines = _normalize_newlines(parsed.replacement).splitlines()
        updated_lines = (
            lines[: parsed.start_line - 1] + replacement_lines + lines[parsed.end_line :]
        )
        newline = _detect_newline(content)
        updated = newline.join(updated_lines)
        if _has_terminal_newline(content) and updated_lines:
            updated += newline
        updated_raw = updated.encode("utf-8")

        def validate() -> None:
            self._read_set.require_current(
                target,
                start_line=parsed.start_line,
                end_line=parsed.end_line,
                expected_hash=parsed.expected_file_hash,
            )

        self._writer.write(target, updated_raw, validate_before_replace=validate)
        self._read_set.refresh_after_write(target, updated_raw)
        before_preview = _preview(selected)
        after_preview = _preview("\n".join(replacement_lines))
        return ToolOutput(
            content=(
                f"Updated {parsed.path} lines {parsed.start_line}-{parsed.end_line}\n"
                f"before: {before_preview}\n"
                f"after: {after_preview}"
            )
        )


def _normalize_newlines(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n")


def _detect_newline(content: str) -> str:
    match = re.search(r"\r\n|\n|\r", content)
    return match.group(0) if match else "\n"


def _has_terminal_newline(content: str) -> bool:
    return content.endswith(("\r\n", "\n", "\r"))


def _preview(content: str, limit: int = 500) -> str:
    compact = "\\n".join(content.splitlines())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}...[truncated]"
