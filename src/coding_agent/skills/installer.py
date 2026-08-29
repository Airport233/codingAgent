from __future__ import annotations

import asyncio
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from coding_agent.skills.loader import SkillLoader, SkillSnapshot


@dataclass(frozen=True, slots=True)
class InstallResult:
    installed: tuple[str, ...]
    message: str
    output: tuple[str, ...] = ()


class SkillInstaller:
    """Thin wrapper around ``npx skills add`` — the de facto standard for
    installing Agent Skills (skills.sh).

    Does not clone or validate skills itself; the ``skills`` CLI handles that
    and writes thin discovery stubs to the project's ``.agents/skills/``
    directory. This class just runs the command, detects what was added,
    streams live output for progress feedback, and offers reload + uninstall.
    """

    def __init__(
        self,
        user_dir: Path,
        project_dir: Path,
        *,
        timeout: float = 120,
    ) -> None:
        self._user_dir = user_dir
        self._project_dir = project_dir
        self._builtin_dir = Path(__file__).parent / "builtin"
        self._timeout = timeout

    async def install(self, source: str) -> AsyncIterator[tuple[str, ...] | InstallResult]:
        """Yield progress lines as tuples, then a final InstallResult."""
        if shutil.which("npx") is None:
            yield InstallResult(
                (),
                "npx is not installed. Install Node.js (https://nodejs.org) to use skill install.",
            )
            return

        before = self._existing_skill_names()
        try:
            process = await asyncio.create_subprocess_exec(
                "npx",
                "-y",
                "skills",
                "add",
                source,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            yield InstallResult((), "npx is not available on PATH.")
            return

        output_lines: list[str] = []
        try:
            assert process.stdout is not None
            async for raw_line in process.stdout:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    output_lines.append(line)
                    yield (line,)
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise

        try:
            await asyncio.wait_for(process.wait(), timeout=self._timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            yield InstallResult(
                (),
                f"npx skills add timed out ({self._timeout:g}s). "
                f"Last output: {output_lines[-1] if output_lines else '(none)'}",
                tuple(output_lines),
            )
            return

        if process.returncode != 0:
            tail = "\n".join(output_lines[-5:])
            yield InstallResult(
                (),
                f"npx skills add failed (exit {process.returncode}).\n{tail}",
                tuple(output_lines),
            )
            return

        after = self._existing_skill_names()
        new = tuple(sorted(after - before))
        if not new:
            yield InstallResult(
                (),
                "Command succeeded but no new skill directories were detected.",
                tuple(output_lines),
            )
            return

        yield InstallResult(
            new,
            f"Installed {len(new)} skill(s): {', '.join(new)}",
            tuple(output_lines),
        )

    async def uninstall(self, name: str) -> bool:
        for directory in (self._project_dir, self._user_dir):
            target = directory / name
            if target.is_dir():
                shutil.rmtree(target)
                return True
        return False

    def reload(self) -> SkillSnapshot:
        return SkillLoader(
            self._builtin_dir,
            user_dir=self._user_dir,
            project_dir=self._project_dir,
        ).load()

    def _existing_skill_names(self) -> set[str]:
        names: set[str] = set()
        for directory in (self._project_dir, self._user_dir):
            if directory.is_dir():
                for entry in directory.iterdir():
                    if entry.is_dir() and (entry / "SKILL.md").is_file():
                        names.add(entry.name)
        return names
