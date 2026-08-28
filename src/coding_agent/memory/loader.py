from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


class MemoryLoadError(Exception):
    """A project memory file cannot be loaded safely."""


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    source: str
    priority: int
    content: str


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    entries: tuple[MemoryEntry, ...]
    digest: str
    rendered: str


class ProjectMemoryLoader:
    filename = "CODING_AGENT.md"

    def __init__(
        self,
        project_root: Path,
        current_directory: Path,
        *,
        max_file_bytes: int = 65_536,
        max_total_bytes: int = 131_072,
    ) -> None:
        if max_file_bytes <= 0 or max_total_bytes <= 0:
            raise ValueError("memory byte limits must be positive")
        self._project_root = project_root.resolve()
        self._current_directory = current_directory.resolve()
        self._max_file_bytes = max_file_bytes
        self._max_total_bytes = max_total_bytes

    def load(self) -> MemorySnapshot:
        if not self._current_directory.is_relative_to(self._project_root):
            raise MemoryLoadError("Current directory must be inside the project")
        if not self._current_directory.is_dir():
            raise MemoryLoadError("Current directory does not exist")

        entries: list[MemoryEntry] = []
        total_bytes = 0
        for priority, directory in enumerate(self._directories_to_current()):
            candidate = directory / self.filename
            if not candidate.exists():
                continue
            resolved = candidate.resolve()
            if not resolved.is_relative_to(self._project_root):
                raise MemoryLoadError(f"Memory file points outside the project: {candidate}")
            if not resolved.is_file():
                raise MemoryLoadError(f"Memory path is not a file: {candidate}")
            if resolved.stat().st_size > self._max_file_bytes:
                raise MemoryLoadError(f"Memory file is too large: {candidate}")
            raw = resolved.read_bytes()
            if len(raw) > self._max_file_bytes:
                raise MemoryLoadError(f"Memory file is too large: {candidate}")
            if b"\x00" in raw:
                raise MemoryLoadError(f"Memory file appears to be binary: {candidate}")
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise MemoryLoadError(f"Memory file is not UTF-8: {candidate}") from error
            total_bytes += len(raw)
            if total_bytes > self._max_total_bytes:
                raise MemoryLoadError("Combined project memory is too large")
            source = candidate.relative_to(self._project_root).as_posix()
            entries.append(MemoryEntry(source, priority, content))

        frozen_entries = tuple(entries)
        return MemorySnapshot(
            entries=frozen_entries,
            digest=_digest_entries(frozen_entries),
            rendered=_render_entries(frozen_entries),
        )

    def _directories_to_current(self) -> tuple[Path, ...]:
        relative = self._current_directory.relative_to(self._project_root)
        directories = [self._project_root]
        current = self._project_root
        for part in relative.parts:
            current /= part
            directories.append(current)
        return tuple(directories)


def _digest_entries(entries: tuple[MemoryEntry, ...]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry.source.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(entry.content.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _render_entries(entries: tuple[MemoryEntry, ...]) -> str:
    if not entries:
        return ""
    sections = ["Project memory follows. Apply all sections; later sections have higher priority."]
    sections.extend(
        f"\n## Memory source: {entry.source} (priority {entry.priority})\n{entry.content}"
        for entry in entries
    )
    return "\n".join(sections)
