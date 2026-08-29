from __future__ import annotations

import asyncio
import shutil
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

    Runs the command, collects output, detects newly added skill directories,
    and returns a result. Does not stream output line-by-line (npx uses block
    buffering on pipes, so streaming is unreliable). Callers should show a
    spinner while waiting and display the collected output on completion.
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

    async def install(self, source: str) -> InstallResult:
        if shutil.which("npx") is None:
            return InstallResult(
                (),
                "npx is not installed. Install Node.js (https://nodejs.org) to use skill install.",
            )

        before = self._existing_skill_names()
        try:
            process = await asyncio.create_subprocess_exec(
                "npx",
                "-y",
                "skills",
                "add",
                source,
                "--all",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            return InstallResult((), "npx is not available on PATH.")

        output_lines: list[str] = []
        assert process.stdout is not None
        try:
            async for raw_line in process.stdout:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    output_lines.append(line)
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise

        try:
            await asyncio.wait_for(process.wait(), timeout=self._timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            tail = output_lines[-1] if output_lines else "(no output)"
            return InstallResult(
                (),
                f"npx skills add timed out ({self._timeout:g}s). Last output: {tail}",
                tuple(output_lines),
            )

        if process.returncode != 0:
            tail = "\n".join(output_lines[-5:])
            return InstallResult(
                (),
                f"npx skills add failed (exit {process.returncode}).\n{tail}",
                tuple(output_lines),
            )

        after = self._existing_skill_names()
        new = tuple(sorted(after - before))
        if not new:
            return InstallResult(
                (),
                "No new skills added — they may already be installed. "
                "Use /skills to see what's available.",
                tuple(output_lines),
            )

        return InstallResult(
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
