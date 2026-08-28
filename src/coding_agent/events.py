from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentStarted:
    prompt: str


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolStarted:
    call_id: str
    tool_name: str


@dataclass(frozen=True, slots=True)
class ToolFinished:
    call_id: str
    tool_name: str
    is_error: bool


@dataclass(frozen=True, slots=True)
class AgentCompleted:
    text: str


@dataclass(frozen=True, slots=True)
class AgentFailed:
    message: str


@dataclass(frozen=True, slots=True)
class WarningRaised:
    message: str


type CoreEvent = (
    AgentStarted
    | TextDelta
    | ToolStarted
    | ToolFinished
    | AgentCompleted
    | AgentFailed
    | WarningRaised
)
