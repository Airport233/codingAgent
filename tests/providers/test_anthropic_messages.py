from __future__ import annotations

from coding_agent.domain import (
    AssistantExchange,
    ThinkingBlock,
    ToolContinuationExchange,
    ToolResultBlock,
    ToolUseBlock,
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
