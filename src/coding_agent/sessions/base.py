from __future__ import annotations

from typing import Protocol


class SessionStore(Protocol):
    async def append(self, kind: str, payload: object) -> None: ...
