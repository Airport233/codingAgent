from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from coding_agent.tools.base import RecoverableToolError


class WorkspaceGuard:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, requested_path: str) -> Path:
        requested = Path(requested_path)
        if requested.is_absolute():
            raise RecoverableToolError("Path is outside the workspace")
        resolved = (self.root / requested).resolve()
        if not resolved.is_relative_to(self.root):
            raise RecoverableToolError("Path is outside the workspace")
        return resolved


@dataclass(frozen=True, slots=True)
class ReadObservation:
    sha256: str
    size: int
    mtime_ns: int
    read_at: datetime
    visible_ranges: tuple[tuple[int, int], ...]


class ReadSet:
    """Tracks file versions and the line ranges that the model has observed."""

    def __init__(self) -> None:
        self._entries: dict[Path, ReadObservation] = {}

    def record(
        self,
        path: Path,
        raw: bytes,
        *,
        start_line: int,
        end_line: int,
        mtime_ns: int,
    ) -> ReadObservation:
        resolved = path.resolve()
        current = self._entries.get(resolved)
        digest = hashlib.sha256(raw).hexdigest()
        visible_ranges = ((start_line, end_line),)
        if current is not None and current.sha256 == digest:
            visible_ranges = current.visible_ranges + visible_ranges
        observation = ReadObservation(
            sha256=digest,
            size=len(raw),
            mtime_ns=mtime_ns,
            read_at=datetime.now(UTC),
            visible_ranges=visible_ranges,
        )
        self._entries[resolved] = observation
        return observation

    def require_current(
        self,
        path: Path,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
        expected_hash: str | None = None,
    ) -> ReadObservation:
        resolved = path.resolve()
        observation = self._entries.get(resolved)
        if observation is None:
            raise RecoverableToolError("File must be read before it can be changed")
        try:
            raw = resolved.read_bytes()
            stat = resolved.stat()
        except OSError as error:
            raise RecoverableToolError("File changed after it was read; read it again") from error
        digest = hashlib.sha256(raw).hexdigest()
        if (
            digest != observation.sha256
            or len(raw) != observation.size
            or stat.st_mtime_ns != observation.mtime_ns
        ):
            raise RecoverableToolError("File changed after it was read; read it again")
        if expected_hash is not None and expected_hash != observation.sha256:
            raise RecoverableToolError("expected_file_hash does not match the read version")
        if start_line is not None and end_line is not None:
            visible = {
                line
                for range_start, range_end in observation.visible_ranges
                for line in range(range_start, range_end + 1)
            }
            if any(line not in visible for line in range(start_line, end_line + 1)):
                raise RecoverableToolError("Requested edit range was not read")
        return observation

    def refresh_after_write(self, path: Path, raw: bytes) -> None:
        resolved = path.resolve()
        current = self._entries.get(resolved)
        if current is None:
            return
        stat = resolved.stat()
        self._entries[resolved] = replace(
            current,
            sha256=hashlib.sha256(raw).hexdigest(),
            size=len(raw),
            mtime_ns=stat.st_mtime_ns,
            read_at=datetime.now(UTC),
            visible_ranges=(),
        )


class AtomicFileWriter:
    """Installs bytes with a same-directory temporary file and atomic replace."""

    def write(
        self,
        target: Path,
        content: bytes,
        *,
        validate_before_replace: Callable[[], None] | None = None,
    ) -> None:
        temporary: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".coding-agent-", suffix=".tmp", dir=target.parent
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            if validate_before_replace is not None:
                validate_before_replace()
            os.replace(temporary, target)
            temporary = None
        except RecoverableToolError:
            raise
        except OSError as error:
            raise RecoverableToolError("Unable to atomically write the file") from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
