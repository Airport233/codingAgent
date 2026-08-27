from __future__ import annotations

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
