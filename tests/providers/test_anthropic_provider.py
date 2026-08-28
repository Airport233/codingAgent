from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from coding_agent.domain import UserExchange
from coding_agent.providers.anthropic import AnthropicMessagesProvider
from coding_agent.providers.base import ProviderResponseFinished, ProviderTextDelta
from coding_agent.tools.base import ToolSpec


class FakeSDKEvent:
    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return self._data


class FakeStream:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self._events = events

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[FakeSDKEvent]:
        async def iterate() -> AsyncIterator[FakeSDKEvent]:
            for event in self._events:
                yield FakeSDKEvent(event)

        return iterate()


class FakeMessages:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.events = events
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        return FakeStream(self.events)


class FakeClient:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.messages = FakeMessages(events)


@pytest.mark.asyncio
async def test_provider_uses_low_level_messages_stream_and_internal_events() -> None:
    client = FakeClient(
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
        ]
    )
    provider = AnthropicMessagesProvider(
        client=client,
        model="example-model",
        max_tokens=2048,
        extra_body={"thinking_effort": "high"},
    )
    tool = ToolSpec(
        name="read_file",
        description="Read a file",
        input_schema={"type": "object", "properties": {}},
    )

    events = [
        event
        async for event in provider.stream(
            (UserExchange(content="Say hello"),),
            (tool,),
            system_instructions="project memory",
        )
    ]

    assert ProviderTextDelta(text="hello") in events
    assert isinstance(events[-1], ProviderResponseFinished)
    assert client.messages.calls == [
        {
            "model": "example-model",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": "Say hello"}],
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "system": "project memory",
            "extra_body": {"thinking_effort": "high"},
        }
    ]


@pytest.mark.asyncio
async def test_provider_ignores_sdk_convenience_events_between_raw_events() -> None:
    client = FakeClient(
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
            {"type": "text", "text": "hello", "snapshot": "hello"},
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2},
            },
            {
                "type": "message_stop",
                "message": {"content": [{"type": "text", "text": "hello"}]},
            },
        ]
    )
    provider = AnthropicMessagesProvider(
        client=client,
        model="example-model",
        max_tokens=128,
    )

    events = [event async for event in provider.stream((UserExchange("hello"),), ())]

    assert events.count(ProviderTextDelta(text="hello")) == 1
    assert isinstance(events[-1], ProviderResponseFinished)
