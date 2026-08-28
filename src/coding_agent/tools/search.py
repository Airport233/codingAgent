from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from coding_agent.tools.base import RecoverableToolError, ToolOutput
from coding_agent.tools.workspace import WorkspaceGuard


class CodeSearchInput(BaseModel):
    query: str = Field(min_length=1)
    regex: bool = False
    glob: str | None = None
    case_sensitive: bool = True
    max_results: int = Field(default=100, ge=1, le=1000)


class CodeSearchTool:
    name = "code_search"
    description = "Search workspace text files using a string or regular expression."
    input_model = CodeSearchInput

    def __init__(self, workspace: Path, *, rg_executable: str | None) -> None:
        self._guard = WorkspaceGuard(workspace)
        self._rg_executable = rg_executable

    async def execute(self, arguments: BaseModel) -> ToolOutput:
        parsed = CodeSearchInput.model_validate(arguments)
        if self._rg_executable:
            try:
                matches = await self._search_with_ripgrep(parsed)
            except OSError:
                matches = self._search_with_python(parsed)
        else:
            matches = self._search_with_python(parsed)
        truncated = len(matches) > parsed.max_results
        visible = matches[: parsed.max_results]
        if truncated:
            visible.append(f"[results truncated at {parsed.max_results} matches]")
        return ToolOutput(content="\n".join(visible) or "No matches.")

    def _search_with_python(self, request: CodeSearchInput) -> list[str]:
        flags = 0 if request.case_sensitive else re.IGNORECASE
        expression = request.query if request.regex else re.escape(request.query)
        try:
            pattern = re.compile(expression, flags)
        except re.error as error:
            raise RecoverableToolError(f"Invalid regular expression: {error}") from error

        matches: list[str] = []
        for path in sorted(self._guard.root.rglob("*")):
            relative = path.relative_to(self._guard.root)
            if not path.is_file() or {".git", ".venv", ".tmp"}.intersection(relative.parts):
                continue
            if request.glob and not relative.match(request.glob):
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                if pattern.search(line):
                    matches.append(f"{relative.as_posix()}:{line_number}:{line}")
                    if len(matches) > request.max_results:
                        return matches
        return matches

    async def _search_with_ripgrep(self, request: CodeSearchInput) -> list[str]:
        if self._rg_executable is None:  # pragma: no cover - guarded by execute
            raise RuntimeError("ripgrep executable is unavailable")
        command = [self._rg_executable, "--json", "--color", "never"]
        if not request.regex:
            command.append("--fixed-strings")
        if not request.case_sensitive:
            command.append("--ignore-case")
        if request.glob:
            command.extend(("--glob", request.glob))
        command.extend(("--", request.query, "."))
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self._guard.root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode not in {0, 1}:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RecoverableToolError(f"ripgrep failed: {message}")
        return self._parse_ripgrep_json(stdout, request.max_results)

    @staticmethod
    def _parse_ripgrep_json(output: bytes, max_results: int) -> list[str]:
        matches: list[str] = []
        for raw_line in output.splitlines():
            event = json.loads(raw_line)
            if event.get("type") != "match":
                continue
            data = event["data"]
            path = data["path"]["text"].replace("\\", "/")
            line_number = data["line_number"]
            line = data["lines"]["text"].rstrip("\r\n")
            matches.append(f"{path}:{line_number}:{line}")
            if len(matches) > max_results:
                break
        return matches
