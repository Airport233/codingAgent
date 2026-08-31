from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from coding_agent.domain import AssistantExchange, ConversationExchange
from coding_agent.tools.base import ToolSpec


class RetryableProviderError(Exception):
    """A response-level failure that is safe to retry before installing an exchange."""

    def __init__(self, message: str, *, code: str = "provider_protocol_error") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ProviderTextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ProviderThinkingDelta:
    thinking: str


@dataclass(frozen=True, slots=True)
class ProviderThinkingSignatureDelta:
    signature: str


@dataclass(frozen=True, slots=True)
class ProviderUsageUpdated:
    usage: dict[str, int]


@dataclass(frozen=True, slots=True)
class ProviderResponseFinished:
    exchange: AssistantExchange


type ProviderEvent = (
    ProviderTextDelta
    | ProviderThinkingDelta
    | ProviderThinkingSignatureDelta
    | ProviderUsageUpdated
    | ProviderResponseFinished
)


class Provider(Protocol):
    def stream(
        self,
        conversation: tuple[ConversationExchange, ...],
        tools: tuple[ToolSpec, ...],
        system_instructions: str | None = None,
    ) -> AsyncIterator[ProviderEvent]: ...
