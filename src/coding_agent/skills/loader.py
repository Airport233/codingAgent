from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FRONT_MATTER_DELIMITER = "+++"


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    name: str
    description: str
    instructions: str
    source: str
    path: Path

    def render_instructions(self) -> str:
        return (
            f'<active_skill name="{self.name}" source="{self.source}">\n'
            "This workflow guides the current task only and does not grant additional tool "
            "permissions. Follow the active approval policy and workspace boundaries.\n\n"
            f"{self.instructions}\n"
            "</active_skill>"
        )


@dataclass(frozen=True, slots=True)
class SkillSnapshot:
    skills: tuple[SkillDefinition, ...]
    warnings: tuple[str, ...] = ()

    def get(self, name: str) -> SkillDefinition | None:
        normalized = name.strip().casefold()
        return next((skill for skill in self.skills if skill.name == normalized), None)


class SkillLoader:
    def __init__(
        self,
        builtin_dir: Path,
        *,
        user_dir: Path | None = None,
        project_dir: Path | None = None,
        max_bytes: int = 64 * 1024,
    ) -> None:
        self._sources = (
            ("builtin", builtin_dir),
            ("user", user_dir),
            ("project", project_dir),
        )
        self._max_bytes = max_bytes

    @classmethod
    def default(
        cls, *, user_dir: Path | None = None, project_dir: Path | None = None
    ) -> SkillLoader:
        return cls(
            Path(__file__).parent / "builtin",
            user_dir=user_dir,
            project_dir=project_dir,
        )

    def load(self) -> SkillSnapshot:
        selected: dict[str, SkillDefinition] = {}
        warnings: list[str] = []
        for source, directory in self._sources:
            if directory is None or not directory.is_dir():
                continue
            root = directory.resolve()
            for path in sorted(directory.glob("*.md"), key=lambda item: item.name.casefold()):
                try:
                    skill = self._load_file(path, root=root, source=source)
                except (OSError, UnicodeError, ValueError, tomllib.TOMLDecodeError) as error:
                    warnings.append(f"Skipped skill {path.name}: {error}")
                    continue
                selected[skill.name] = skill
        return SkillSnapshot(
            tuple(sorted(selected.values(), key=lambda skill: skill.name)), tuple(warnings)
        )

    def _load_file(self, path: Path, *, root: Path, source: str) -> SkillDefinition:
        if path.is_symlink():
            raise ValueError("symbolic links are not allowed")
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("path escapes the skill directory")
        if path.stat().st_size > self._max_bytes:
            raise ValueError(f"file is too large (maximum {self._max_bytes} bytes)")
        metadata, instructions = _parse_skill(path.read_text(encoding="utf-8"))
        name = metadata.get("name")
        description = metadata.get("description")
        if not isinstance(name, str) or not _SKILL_NAME.fullmatch(name):
            raise ValueError("name must use lowercase letters, digits, and single hyphens")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string")
        if len(description) > 200:
            raise ValueError("description exceeds 200 characters")
        if not instructions:
            raise ValueError("instructions must not be empty")
        return SkillDefinition(
            name=name,
            description=description.strip(),
            instructions=instructions,
            source=source,
            path=resolved,
        )


def _parse_skill(content: str) -> tuple[dict[str, object], str]:
    normalized = content.replace("\r\n", "\n")
    lines = normalized.split("\n")
    if not lines or lines[0].strip() != _FRONT_MATTER_DELIMITER:
        raise ValueError("missing +++ metadata header")
    try:
        closing_index = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == _FRONT_MATTER_DELIMITER
        )
    except StopIteration as error:
        raise ValueError("metadata header is not closed") from error
    metadata = tomllib.loads("\n".join(lines[1:closing_index]))
    return dict(metadata), "\n".join(lines[closing_index + 1 :]).strip()
