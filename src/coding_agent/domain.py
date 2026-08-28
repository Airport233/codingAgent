from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    thinking: str
    signature: str | None = None
    raw: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RedactedThinkingBlock:
    data: str
    raw: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    call_id: str
    name: str
    input: dict[str, object]
    raw: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UnknownProviderBlock:
    block_type: str
    raw: dict[str, object]


type ContentBlock = (
    TextBlock | ThinkingBlock | RedactedThinkingBlock | ToolUseBlock | UnknownProviderBlock
)
type StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal"]


@dataclass(frozen=True, slots=True)
class UserExchange:
    content: str


@dataclass(frozen=True, slots=True)
class AssistantExchange:
    blocks: tuple[ContentBlock, ...]
    stop_reason: StopReason
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.blocks if isinstance(block, TextBlock))

    @property
    def tool_uses(self) -> tuple[ToolUseBlock, ...]:
        return tuple(block for block in self.blocks if isinstance(block, ToolUseBlock))


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolContinuationExchange:
    assistant: AssistantExchange
    results: tuple[ToolResultBlock, ...]

    def __post_init__(self) -> None:
        call_ids = tuple(call.call_id for call in self.assistant.tool_uses)
        result_ids = tuple(result.tool_use_id for result in self.results)
        if call_ids != result_ids:
            raise ValueError("tool results must match assistant tool calls in order")


type ConversationExchange = UserExchange | AssistantExchange | ToolContinuationExchange


@dataclass(slots=True)
class Conversation:
    exchanges: list[ConversationExchange] = field(default_factory=list)

    def snapshot(self) -> tuple[ConversationExchange, ...]:
        return tuple(self.exchanges)
