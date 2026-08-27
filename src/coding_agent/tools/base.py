from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, object]


@dataclass(frozen=True, slots=True)
class ToolOutput:
    content: str


class Tool(Protocol):
    name: str
    description: str

    @property
    def input_model(self) -> type[BaseModel]: ...

    async def execute(self, arguments: BaseModel) -> ToolOutput: ...


class ToolSource(Protocol):
    source_id: str

    async def list_tools(self) -> tuple[Tool, ...]: ...


class RecoverableToolError(Exception):
    """An expected tool failure that should be returned to the model."""
