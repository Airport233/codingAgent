from __future__ import annotations

from copy import deepcopy

from coding_agent.domain import (
    AssistantExchange,
    ConversationExchange,
    ProviderContinuationExchange,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolContinuationExchange,
    ToolUseBlock,
    UnknownProviderBlock,
    UserExchange,
)


def encode_conversation(
    conversation: tuple[ConversationExchange, ...],
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    for exchange in conversation:
        if isinstance(exchange, UserExchange):
            messages.append({"role": "user", "content": exchange.content})
        elif isinstance(exchange, AssistantExchange):
            messages.append(_encode_assistant(exchange))
        elif isinstance(exchange, ToolContinuationExchange):
            messages.append(_encode_assistant(exchange.assistant))
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": result.tool_use_id,
                            "content": result.content,
                            "is_error": result.is_error,
                        }
                        for result in exchange.results
                    ],
                }
            )
        elif isinstance(exchange, ProviderContinuationExchange):
            messages.append(_encode_assistant(exchange.assistant))
            messages.append({"role": "user", "content": exchange.instruction})
    return messages


def _encode_assistant(exchange: AssistantExchange) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": [_encode_block(block) for block in exchange.blocks],
    }


def _encode_block(block: object) -> dict[str, object]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingBlock):
        if block.raw:
            return deepcopy(block.raw)
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "signature": block.signature or "",
        }
    if isinstance(block, RedactedThinkingBlock):
        return (
            deepcopy(block.raw) if block.raw else {"type": "redacted_thinking", "data": block.data}
        )
    if isinstance(block, ToolUseBlock):
        if block.raw:
            return deepcopy(block.raw)
        return {
            "type": "tool_use",
            "id": block.call_id,
            "name": block.name,
            "input": deepcopy(block.input),
        }
    if isinstance(block, UnknownProviderBlock):
        return deepcopy(block.raw)
    raise TypeError(f"unsupported assistant content block: {type(block).__name__}")
