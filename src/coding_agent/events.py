from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentStarted:
    prompt: str


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ThinkingStarted:
    pass


@dataclass(frozen=True, slots=True)
class ThinkingDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ThinkingFinished:
    pass


@dataclass(frozen=True, slots=True)
class ContextUsageChanged:
    used_tokens: int
    context_window: int
    level: str


@dataclass(frozen=True, slots=True)
class ToolStarted:
    call_id: str
    tool_name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class ToolFinished:
    call_id: str
    tool_name: str
    is_error: bool
    content: str
    metadata: dict[str, object]

    @property
    def status(self) -> str:
        if self.metadata.get("cancelled") is True:
            return "cancelled"
        if self.metadata.get("timed_out") is True:
            return "timeout"
        return "error" if self.is_error else "done"


@dataclass(frozen=True, slots=True)
class AgentCompleted:
    text: str


@dataclass(frozen=True, slots=True)
class AgentFailed:
    message: str


@dataclass(frozen=True, slots=True)
class WarningRaised:
    message: str


@dataclass(frozen=True, slots=True)
class AgentCancelled:
    message: str


type CoreEvent = (
    AgentStarted
    | TextDelta
    | ThinkingStarted
    | ThinkingDelta
    | ThinkingFinished
    | ContextUsageChanged
    | ToolStarted
    | ToolFinished
    | AgentCompleted
    | AgentFailed
    | AgentCancelled
    | WarningRaised
)
