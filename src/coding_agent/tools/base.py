from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, object]


@dataclass(frozen=True, slots=True)
class ToolOutput:
    content: str
    metadata: dict[str, object] = field(default_factory=dict)


class Tool(Protocol):
    name: str
    description: str

    @property
    def input_model(self) -> type[BaseModel]: ...

    async def execute(self, arguments: BaseModel) -> ToolOutput: ...


type ToolOutputStream = Literal["stdout", "stderr"]
type ToolOutputCallback = Callable[[ToolOutputStream, str], None]


@runtime_checkable
class StreamingTool(Protocol):
    async def execute_with_output(
        self, arguments: BaseModel, on_output: ToolOutputCallback
    ) -> ToolOutput: ...


class ToolSource(Protocol):
    source_id: str

    async def list_tools(self) -> tuple[Tool, ...]: ...


class RecoverableToolError(Exception):
    """An expected tool failure that should be returned to the model."""
