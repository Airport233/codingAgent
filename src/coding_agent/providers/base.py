from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from coding_agent.domain import AssistantExchange, ConversationExchange
from coding_agent.tools.base import ToolSpec


@dataclass(frozen=True, slots=True)
class ProviderTextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ProviderResponseFinished:
    exchange: AssistantExchange


type ProviderEvent = ProviderTextDelta | ProviderResponseFinished


class Provider(Protocol):
    def stream(
        self,
        conversation: tuple[ConversationExchange, ...],
        tools: tuple[ToolSpec, ...],
    ) -> AsyncIterator[ProviderEvent]: ...
