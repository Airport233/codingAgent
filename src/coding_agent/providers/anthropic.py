from __future__ import annotations

from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any, cast

from coding_agent.domain import ConversationExchange
from coding_agent.providers.anthropic_messages import encode_conversation
from coding_agent.providers.anthropic_stream import AnthropicStreamAggregator
from coding_agent.providers.base import ProviderEvent
from coding_agent.tools.base import ToolSpec

_SDK_CONVENIENCE_EVENT_TYPES = frozenset(
    {"citation", "input_json", "signature", "text", "thinking"}
)


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
        supports_tools: bool = True,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._extra_body = deepcopy(extra_body or {})
        self._supports_tools = supports_tools

    async def stream(
        self,
        conversation: tuple[ConversationExchange, ...],
        tools: tuple[ToolSpec, ...],
        system_instructions: str | None = None,
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
        request: dict[str, object] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": encode_conversation(conversation),
            "extra_body": deepcopy(self._extra_body),
        }
        if self._supports_tools:
            request["tools"] = request_tools
        if system_instructions:
            request["system"] = system_instructions
        async with self._client.messages.stream(**request) as sdk_stream:
            async for sdk_event in sdk_stream:
                raw = cast(dict[str, object], sdk_event.model_dump(mode="json"))
                if raw.get("type") in _SDK_CONVENIENCE_EVENT_TYPES:
                    continue
                for event in aggregator.consume(raw):
                    yield event
