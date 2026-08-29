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


class SkillInstaller:
    """Thin wrapper around ``npx skills add`` — the de facto standard for
    installing Agent Skills (skills.sh).

    Does not clone or validate skills itself; the ``skills`` CLI handles that
    and writes thin discovery stubs to the project's ``.claude/skills/``
    directory. This class just runs the command, detects what was added,
    and offers a reload + uninstall path.
    """

    def __init__(self, user_dir: Path, project_dir: Path) -> None:
        self._user_dir = user_dir
        self._project_dir = project_dir
        self._builtin_dir = Path(__file__).parent / "builtin"
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
                "skills",
                "add",
                source,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return InstallResult((), "npx is not available on PATH.")
        try:
            await asyncio.wait_for(process.wait(), timeout=60)
        except TimeoutError:
            process.kill()
            await process.wait()
            return InstallResult((), "npx skills add timed out (60s).")
        if process.returncode != 0:
            stderr = b""
            if process.stderr is not None:
                stderr = await process.stderr.read()
            return InstallResult(
                (),
                f"npx skills add failed: {stderr.decode('utf-8', errors='replace').strip()}",
            )
        after = self._existing_skill_names()
        new = tuple(sorted(after - before))
        if not new:
            return InstallResult(
                (),
                "Command succeeded but no new skill directories were detected.",
            )
        return InstallResult(new, f"Installed {len(new)} skill(s): {', '.join(new)}")

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
