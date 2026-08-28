from __future__ import annotations

from coding_agent.domain import (
    AssistantExchange,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolContinuationExchange,
    ToolResultBlock,
    ToolUseBlock,
    UnknownProviderBlock,
    UserExchange,
)
from coding_agent.providers.anthropic_messages import encode_conversation


def test_tool_continuation_round_trips_thinking_and_orders_results_first() -> None:
    thinking_raw = {
        "type": "thinking",
        "thinking": "inspect",
        "signature": "opaque-signature",
    }
    tool_raw = {
        "type": "tool_use",
        "id": "call-1",
        "name": "read_file",
        "input": {"path": "hello.txt"},
    }
    assistant = AssistantExchange(
        blocks=(
            ThinkingBlock(
                thinking="inspect",
                signature="opaque-signature",
                raw=thinking_raw,
            ),
            ToolUseBlock(
                call_id="call-1",
                name="read_file",
                input={"path": "hello.txt"},
                raw=tool_raw,
            ),
        ),
        stop_reason="tool_use",
    )
    continuation = ToolContinuationExchange(
        assistant=assistant,
        results=(
            ToolResultBlock(
                tool_use_id="call-1",
                content="1: hello",
                is_error=False,
            ),
        ),
    )

    messages = encode_conversation((UserExchange("Read hello.txt"), continuation))

    assert messages == [
        {"role": "user", "content": "Read hello.txt"},
        {"role": "assistant", "content": [thinking_raw, tool_raw]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-1",
                    "content": "1: hello",
                    "is_error": False,
                }
            ],
        },
    ]


def test_encodes_assistant_block_variants_without_mutating_raw_data() -> None:
    redacted_raw = {"type": "redacted_thinking", "data": "opaque"}
    unknown_raw = {"type": "future_block", "payload": {"value": 1}}
    exchange = AssistantExchange(
        blocks=(
            TextBlock("hello"),
            ThinkingBlock("reason", "signature"),
            RedactedThinkingBlock("opaque", raw=redacted_raw),
            ToolUseBlock("call-2", "read_file", {"path": "a.txt"}),
            UnknownProviderBlock("future_block", raw=unknown_raw),
        ),
        stop_reason="end_turn",
    )

    messages = encode_conversation((exchange,))

    assert messages[0]["content"] == [
        {"type": "text", "text": "hello"},
        {"type": "thinking", "thinking": "reason", "signature": "signature"},
        {"type": "redacted_thinking", "data": "opaque"},
        {
            "type": "tool_use",
            "id": "call-2",
            "name": "read_file",
            "input": {"path": "a.txt"},
        },
        unknown_raw,
    ]
    assert redacted_raw == {"type": "redacted_thinking", "data": "opaque"}
    assert unknown_raw == {"type": "future_block", "payload": {"value": 1}}
