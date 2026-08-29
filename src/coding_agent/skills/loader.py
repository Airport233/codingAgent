from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

import yaml

_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FRONT_MATTER_DELIMITER = "---"
_ENTRYPOINT = "SKILL.md"


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    name: str
    description: str
    source: str
    path: Path
    root: Path
    max_bytes: int = field(repr=False)
    max_resource_bytes: int = field(default=256 * 1024, repr=False)
    license: str | None = None
    compatibility: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()
    allowed_tools: str | None = None

    def render_instructions(self) -> str:
        metadata, instructions = _parse_skill(self._read_entrypoint())
        _validate_metadata(metadata, directory_name=self.root.name)
        if not instructions:
            raise ValueError("instructions must not be empty")
        return (
            f'<active_skill name="{self.name}" source="{self.source}">\n'
            "This skill guides the current task only and does not grant additional tool "
            "permissions. Follow the active approval policy and workspace boundaries. "
            "Load referenced files only when needed with read_skill_resource, using paths "
            "relative to this skill directory.\n\n"
            f"{instructions}\n"
            "</active_skill>"
        )

    def read_resource(self, relative_path: str) -> str:
        relative = Path(relative_path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("resource path must stay inside the skill directory")
        if relative.as_posix().casefold() == _ENTRYPOINT.casefold():
            raise ValueError("SKILL.md is loaded by skill activation, not as a resource")
        candidate = self.root
        for part in relative.parts:
            candidate /= part
            if candidate.is_symlink():
                raise ValueError("symbolic links are not allowed in skill resources")
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("resource path escapes the skill directory")
        if not resolved.is_file():
            raise ValueError("skill resource does not exist")
        if resolved.stat().st_size > self.max_resource_bytes:
            raise ValueError(
                f"skill resource is too large (maximum {self.max_resource_bytes} bytes)"
            )
        try:
            return resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("skill resource is not UTF-8 text") from error

    def _read_entrypoint(self) -> str:
        if self.path.is_symlink():
            raise ValueError("symbolic links are not allowed")
        resolved = self.path.resolve()
        if not resolved.is_relative_to(self.root) or resolved.parent != self.root:
            raise ValueError("SKILL.md escapes the skill directory")
        if resolved.stat().st_size > self.max_bytes:
            raise ValueError(f"SKILL.md is too large (maximum {self.max_bytes} bytes)")
        return resolved.read_text(encoding="utf-8")


@dataclass(frozen=True, slots=True)
class SkillSnapshot:
    skills: tuple[SkillDefinition, ...]
    warnings: tuple[str, ...] = ()

    def get(self, name: str) -> SkillDefinition | None:
        normalized = name.strip().casefold()
        return next((skill for skill in self.skills if skill.name == normalized), None)

    def render_catalog(self) -> str:
        if not self.skills:
            return ""
        lines = [
            "<available_skills>",
            "Only skill metadata is listed here. Call activate_skill before following a "
            "skill's instructions.",
        ]
        for skill in self.skills:
            lines.extend(
                (
                    "<skill>",
                    f"<name>{escape(skill.name)}</name>",
                    f"<description>{escape(skill.description)}</description>",
                    f"<location>{escape(str(skill.path))}</location>",
                    f"<source>{escape(skill.source)}</source>",
                    "</skill>",
                )
            )
        lines.append("</available_skills>")
        return "\n".join(lines)


class SkillLoader:
    def __init__(
        self,
        builtin_dir: Path,
        *,
        user_dir: Path | None = None,
        project_dir: Path | None = None,
        max_bytes: int = 64 * 1024,
        max_metadata_bytes: int = 16 * 1024,
        max_resource_bytes: int = 256 * 1024,
    ) -> None:
        self._sources = (
            ("builtin", builtin_dir),
            ("user", user_dir),
            ("project", project_dir),
        )
        self._max_bytes = max_bytes
        self._max_metadata_bytes = max_metadata_bytes
        self._max_resource_bytes = max_resource_bytes

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
            source_root = directory.resolve()
            for skill_dir in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
                if not skill_dir.is_dir() and not skill_dir.is_symlink():
                    if skill_dir.suffix.casefold() == ".md":
                        warnings.append(
                            f"Skipped skill {skill_dir.name}: legacy single-file skills are "
                            "unsupported; use <skill-name>/SKILL.md"
                        )
                    continue
                try:
                    skill = self._load_directory(skill_dir, source_root=source_root, source=source)
                except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
                    warnings.append(f"Skipped skill {skill_dir.name}: {error}")
                    continue
                selected[skill.name] = skill
        return SkillSnapshot(
            tuple(sorted(selected.values(), key=lambda skill: skill.name)), tuple(warnings)
        )

    def _load_directory(
        self, skill_dir: Path, *, source_root: Path, source: str
    ) -> SkillDefinition:
        if skill_dir.is_symlink():
            raise ValueError("symbolic links are not allowed")
        root = skill_dir.resolve()
        if not root.is_relative_to(source_root):
            raise ValueError("path escapes the skills directory")
        entrypoint = skill_dir / _ENTRYPOINT
        if not entrypoint.is_file():
            raise ValueError("missing SKILL.md")
        if entrypoint.is_symlink():
            raise ValueError("symbolic links are not allowed")
        if entrypoint.stat().st_size > self._max_bytes:
            raise ValueError(f"SKILL.md is too large (maximum {self._max_bytes} bytes)")
        metadata = _read_metadata(entrypoint, self._max_metadata_bytes)
        validated = _validate_metadata(metadata, directory_name=skill_dir.name)
        extra_metadata = validated.get("metadata", {})
        assert isinstance(extra_metadata, dict)
        license_value = validated.get("license")
        compatibility_value = validated.get("compatibility")
        allowed_tools_value = validated.get("allowed-tools")
        return SkillDefinition(
            name=str(validated["name"]),
            description=str(validated["description"]),
            source=source,
            path=entrypoint.resolve(),
            root=root,
            max_bytes=self._max_bytes,
            max_resource_bytes=self._max_resource_bytes,
            license=license_value if isinstance(license_value, str) else None,
            compatibility=(compatibility_value if isinstance(compatibility_value, str) else None),
            metadata=tuple(sorted((str(key), str(value)) for key, value in extra_metadata.items())),
            allowed_tools=(allowed_tools_value if isinstance(allowed_tools_value, str) else None),
        )


def _read_metadata(path: Path, max_bytes: int) -> dict[str, object]:
    with path.open("rb") as handle:
        first = handle.readline(max_bytes + 1)
        if first.strip() != _FRONT_MATTER_DELIMITER.encode():
            raise ValueError("missing YAML metadata header")
        consumed = len(first)
        lines: list[bytes] = []
        while consumed <= max_bytes:
            line = handle.readline(max_bytes - consumed + 1)
            if not line:
                raise ValueError("metadata header is not closed")
            consumed += len(line)
            if consumed > max_bytes:
                break
            if line.strip() == _FRONT_MATTER_DELIMITER.encode():
                loaded = yaml.safe_load(b"".join(lines).decode("utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError("YAML metadata must be a mapping")
                return {str(key): value for key, value in loaded.items()}
            lines.append(line)
    raise ValueError(f"metadata header is too large (maximum {max_bytes} bytes)")


def _parse_skill(content: str) -> tuple[dict[str, object], str]:
    normalized = content.replace("\r\n", "\n")
    lines = normalized.split("\n")
    if not lines or lines[0].strip() != _FRONT_MATTER_DELIMITER:
        raise ValueError("missing YAML metadata header")
    try:
        closing_index = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == _FRONT_MATTER_DELIMITER
        )
    except StopIteration as error:
        raise ValueError("metadata header is not closed") from error
    try:
        metadata = yaml.safe_load("\n".join(lines[1:closing_index]))
    except yaml.YAMLError as error:
        raise ValueError("invalid YAML metadata") from error
    if not isinstance(metadata, dict):
        raise ValueError("YAML metadata must be a mapping")
    return {str(key): value for key, value in metadata.items()}, "\n".join(
        lines[closing_index + 1 :]
    ).strip()


def _validate_metadata(metadata: dict[str, object], *, directory_name: str) -> dict[str, object]:
    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or not _SKILL_NAME.fullmatch(name) or len(name) > 64:
        raise ValueError("name must use 1-64 lowercase letters, digits, and single hyphens")
    if name != directory_name:
        raise ValueError("name must match the parent directory")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description must be a non-empty string")
    if len(description) > 1024:
        raise ValueError("description exceeds 1024 characters")
    validated = dict(metadata)
    validated["description"] = description.strip()
    for field_name, limit in (("license", None), ("compatibility", 500), ("allowed-tools", None)):
        value = validated.get(field_name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")
        if isinstance(value, str) and limit is not None and not 1 <= len(value) <= limit:
            raise ValueError(f"{field_name} must contain 1-{limit} characters")
    extra_metadata = validated.get("metadata", {})
    if not isinstance(extra_metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in extra_metadata.items()
    ):
        raise ValueError("metadata must map strings to strings")
    validated["metadata"] = extra_metadata
    return validated
