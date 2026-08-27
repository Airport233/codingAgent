from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SessionRecord:
    kind: str
    payload: object


@dataclass(slots=True)
class InMemorySessionStore:
    records: list[SessionRecord] = field(default_factory=list)

    @property
    def kinds(self) -> list[str]:
        return [record.kind for record in self.records]

    async def append(self, kind: str, payload: object) -> None:
        self.records.append(SessionRecord(kind=kind, payload=payload))
