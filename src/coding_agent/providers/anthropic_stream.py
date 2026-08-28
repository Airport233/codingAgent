from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import cast

from coding_agent.domain import (
    AssistantExchange,
    ContentBlock,
    RedactedThinkingBlock,
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    UnknownProviderBlock,
)
from coding_agent.providers.base import (
    ProviderEvent,
    ProviderResponseFinished,
    ProviderTextDelta,
    ProviderThinkingDelta,
    ProviderThinkingSignatureDelta,
    ProviderUsageUpdated,
)


class AnthropicProtocolError(ValueError):
    """Raised when a stream cannot be converted without inventing protocol data."""


@dataclass(slots=True)
class _BlockSlot:
    block_type: str
    raw: dict[str, object]
    text_parts: list[str] = field(default_factory=list)
    thinking_parts: list[str] = field(default_factory=list)
    signature_parts: list[str] = field(default_factory=list)
    input_json_parts: list[str] = field(default_factory=list)


class AnthropicStreamAggregator:
    def __init__(self) -> None:
        self._open_slots: dict[int, _BlockSlot] = {}
        self._blocks: dict[int, ContentBlock] = {}
        self._usage: dict[str, int] = {}
        self._stop_reason: StopReason | None = None

    def consume(self, event: dict[str, object]) -> tuple[ProviderEvent, ...]:
        event_type = self._string(event, "type")
        if event_type == "message_start":
            message = self._mapping(event, "message")
            self._merge_usage(self._mapping(message, "usage"))
            return ()
        if event_type == "content_block_start":
            self._start_block(event)
            return ()
        if event_type == "content_block_delta":
            return self._apply_delta(event)
        if event_type == "content_block_stop":
            self._stop_block(event)
            return ()
        if event_type == "message_delta":
            delta = self._mapping(event, "delta")
            raw_reason = delta.get("stop_reason")
            if raw_reason is not None:
                self._stop_reason = self._normalize_stop_reason(str(raw_reason))
            usage = self._mapping(event, "usage")
            self._merge_usage(usage)
            return (ProviderUsageUpdated(usage=dict(self._usage)),)
        if event_type == "message_stop":
            if self._open_slots:
                raise AnthropicProtocolError("message stopped with unfinished content blocks")
            if self._stop_reason is None:
                raise AnthropicProtocolError("message stopped without a stop reason")
            blocks = tuple(self._blocks[index] for index in sorted(self._blocks))
            return (
                ProviderResponseFinished(
                    exchange=AssistantExchange(
                        blocks=blocks,
                        stop_reason=self._stop_reason,
                        usage=dict(self._usage),
                    )
                ),
            )
        if event_type == "ping":
            return ()
        raise AnthropicProtocolError(f"unknown stream event type: {event_type}")

    def _start_block(self, event: dict[str, object]) -> None:
        index = self._index(event)
        if index in self._open_slots or index in self._blocks:
            raise AnthropicProtocolError(f"duplicate content block index: {index}")
        raw = deepcopy(self._mapping(event, "content_block"))
        block_type = self._string(raw, "type")
        slot = _BlockSlot(block_type=block_type, raw=raw)
        if block_type == "text":
            slot.text_parts.append(str(raw.get("text", "")))
        elif block_type == "thinking":
            slot.thinking_parts.append(str(raw.get("thinking", "")))
            slot.signature_parts.append(str(raw.get("signature", "")))
        self._open_slots[index] = slot

    def _apply_delta(self, event: dict[str, object]) -> tuple[ProviderEvent, ...]:
        slot = self._slot(event)
        delta = self._mapping(event, "delta")
        delta_type = self._string(delta, "type")
        if delta_type == "text_delta" and slot.block_type == "text":
            text = self._string(delta, "text")
            slot.text_parts.append(text)
            return (ProviderTextDelta(text=text),)
        if delta_type == "thinking_delta" and slot.block_type == "thinking":
            thinking = self._string(delta, "thinking")
            slot.thinking_parts.append(thinking)
            return (ProviderThinkingDelta(thinking=thinking),)
        if delta_type == "signature_delta" and slot.block_type == "thinking":
            signature = self._string(delta, "signature")
            slot.signature_parts.append(signature)
            return (ProviderThinkingSignatureDelta(signature=signature),)
        if delta_type == "input_json_delta" and slot.block_type == "tool_use":
            slot.input_json_parts.append(self._string(delta, "partial_json"))
            return ()
        raise AnthropicProtocolError(
            f"delta {delta_type} is invalid for content block {slot.block_type}"
        )

    def _stop_block(self, event: dict[str, object]) -> None:
        index = self._index(event)
        try:
            slot = self._open_slots.pop(index)
        except KeyError as error:
            raise AnthropicProtocolError(f"unknown content block index: {index}") from error
        raw = deepcopy(slot.raw)
        if slot.block_type == "text":
            text = "".join(slot.text_parts)
            raw["text"] = text
            block: ContentBlock = TextBlock(text=text)
        elif slot.block_type == "thinking":
            thinking = "".join(slot.thinking_parts)
            signature = "".join(slot.signature_parts)
            raw.update(thinking=thinking, signature=signature)
            block = ThinkingBlock(thinking=thinking, signature=signature, raw=raw)
        elif slot.block_type == "redacted_thinking":
            block = RedactedThinkingBlock(data=self._string(raw, "data"), raw=raw)
        elif slot.block_type == "tool_use":
            block = self._finish_tool_use(slot, raw)
        else:
            block = UnknownProviderBlock(block_type=slot.block_type, raw=raw)
        self._blocks[index] = block

    def _finish_tool_use(self, slot: _BlockSlot, raw: dict[str, object]) -> ToolUseBlock:
        call_id = self._string(raw, "id")
        name = self._string(raw, "name")
        if slot.input_json_parts:
            try:
                parsed = json.loads("".join(slot.input_json_parts))
            except json.JSONDecodeError as error:
                raise AnthropicProtocolError("invalid tool input JSON") from error
        else:
            parsed = raw.get("input", {})
        if not isinstance(parsed, dict):
            raise AnthropicProtocolError("tool input must be a JSON object")
        tool_input = cast(dict[str, object], parsed)
        raw["input"] = deepcopy(tool_input)
        return ToolUseBlock(
            call_id=call_id,
            name=name,
            input=tool_input,
            raw=raw,
        )

    def _slot(self, event: dict[str, object]) -> _BlockSlot:
        index = self._index(event)
        try:
            return self._open_slots[index]
        except KeyError as error:
            raise AnthropicProtocolError(f"unknown content block index: {index}") from error

    def _merge_usage(self, usage: dict[str, object]) -> None:
        for key, value in usage.items():
            if isinstance(value, int):
                self._usage[key] = value

    @staticmethod
    def _normalize_stop_reason(reason: str) -> StopReason:
        if reason in {"end_turn", "stop_sequence"}:
            return "end_turn"
        if reason in {"tool_use", "max_tokens", "refusal"}:
            return cast(StopReason, reason)
        raise AnthropicProtocolError(f"unsupported stop reason: {reason}")

    @staticmethod
    def _mapping(container: dict[str, object], key: str) -> dict[str, object]:
        value = container.get(key)
        if not isinstance(value, dict):
            raise AnthropicProtocolError(f"{key} must be an object")
        return value

    @staticmethod
    def _string(container: dict[str, object], key: str) -> str:
        value = container.get(key)
        if not isinstance(value, str) or not value:
            if value == "" and key in {"partial_json", "signature", "text", "thinking"}:
                return ""
            raise AnthropicProtocolError(f"{key} must be a string")
        return value

    @staticmethod
    def _index(event: dict[str, object]) -> int:
        index = event.get("index")
        if not isinstance(index, int) or index < 0:
            raise AnthropicProtocolError("content block index must be a non-negative integer")
        return index
