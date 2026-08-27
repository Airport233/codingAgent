from __future__ import annotations

import pytest

from coding_agent.domain import ThinkingBlock, ToolUseBlock
from coding_agent.providers.anthropic_stream import (
    AnthropicProtocolError,
    AnthropicStreamAggregator,
)
from coding_agent.providers.base import ProviderResponseFinished, ProviderTextDelta


def feed(aggregator: AnthropicStreamAggregator, events: list[dict[str, object]]):
    return [output for event in events for output in aggregator.consume(event)]


def test_aggregates_thinking_signature_and_partial_tool_json() -> None:
    aggregator = AnthropicStreamAggregator()
    outputs = feed(
        aggregator,
        [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "inspect carefully"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "opaque-signature"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "read_file",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"path":'},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '"hello.txt"}'},
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 12},
            },
            {"type": "message_stop"},
        ],
    )

    finished = next(output for output in outputs if isinstance(output, ProviderResponseFinished))
    thinking, tool_use = finished.exchange.blocks
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.thinking == "inspect carefully"
    assert thinking.signature == "opaque-signature"
    assert thinking.raw == {
        "type": "thinking",
        "thinking": "inspect carefully",
        "signature": "opaque-signature",
    }
    assert isinstance(tool_use, ToolUseBlock)
    assert tool_use.call_id == "call-1"
    assert tool_use.input == {"path": "hello.txt"}
    assert tool_use.raw["input"] == {"path": "hello.txt"}
    assert finished.exchange.stop_reason == "tool_use"
    assert finished.exchange.usage == {"output_tokens": 12}


def test_text_delta_is_forwarded_and_preserved() -> None:
    aggregator = AnthropicStreamAggregator()
    outputs = feed(
        aggregator,
        [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        ],
    )

    assert ProviderTextDelta(text="hello") in outputs
    finished = next(output for output in outputs if isinstance(output, ProviderResponseFinished))
    assert finished.exchange.text == "hello"


def test_tool_json_is_parsed_only_when_block_stops() -> None:
    aggregator = AnthropicStreamAggregator()
    aggregator.consume(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "call-1",
                "name": "read_file",
                "input": {},
            },
        }
    )
    aggregator.consume(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{"},
        }
    )

    with pytest.raises(AnthropicProtocolError, match="invalid tool input JSON"):
        aggregator.consume({"type": "content_block_stop", "index": 0})
