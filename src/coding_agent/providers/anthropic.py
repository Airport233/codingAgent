from __future__ import annotations

from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any, cast

from coding_agent.domain import ConversationExchange
from coding_agent.providers.anthropic_messages import encode_conversation
from coding_agent.providers.anthropic_stream import AnthropicStreamAggregator
from coding_agent.providers.base import ProviderEvent
from coding_agent.tools.base import ToolSpec


class AnthropicMessagesProvider:
    """Low-level Anthropic Messages stream adapter.

    The SDK object is deliberately contained at this boundary. Core modules only
    receive repository-owned ProviderEvent values.
    """

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_tokens: int,
        extra_body: dict[str, object] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._extra_body = deepcopy(extra_body or {})

    async def stream(
        self,
        conversation: tuple[ConversationExchange, ...],
        tools: tuple[ToolSpec, ...],
    ) -> AsyncIterator[ProviderEvent]:
        aggregator = AnthropicStreamAggregator()
        request_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": deepcopy(tool.input_schema),
            }
            for tool in tools
        ]
        async with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=encode_conversation(conversation),
            tools=request_tools,
            extra_body=deepcopy(self._extra_body),
        ) as sdk_stream:
            async for sdk_event in sdk_stream:
                raw = cast(dict[str, object], sdk_event.model_dump(mode="json"))
                for event in aggregator.consume(raw):
                    yield event
